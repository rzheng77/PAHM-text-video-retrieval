###########################################
#####  adapt from X-CLIP (https://arxiv.org/abs/2207.07285), thanks a lot! ###########
###########################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.until_module import PreTrainedModel, AllGather, CrossEn, make_patch_shift, MILNCELoss_BoF
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from modules.modeling import CLIP4ClipPreTrainedModel, show_log, update_attr, check_attr
from modules.differential_topk import VisualTokenSelection, TextTokenSelection, VisualTokenRandomSelection, STVisualTokenSelection
from modules.cluster.fast_kmeans import batch_fast_kmedoids_with_split
import numpy as np

logger = logging.getLogger(__name__)
allgather = AllGather.apply


class My(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(My, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        num_words = task_config.max_words
        num_frames = self.task_config.max_frames

        # recommend set True
        self.use_original_clip_for_frame_features = True    

        # for coarse-grained constrast weights
        self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)    
        self.instance_mat_weight = nn.parameter.Parameter(torch.eye(24), requires_grad=True)
        self.object_mat_weight = nn.parameter.Parameter(torch.eye(4), requires_grad=True)
        self.instance_mat_weight1 = nn.parameter.Parameter(torch.eye(8), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(8), requires_grad=True)    

        num_frames = self.task_config.max_frames
        num_patch = 4 # hyperparameter

        self.patch_feattype_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, 12-1), nn.GELU())

        self.text_feattype_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, 24), nn.GELU())
        self.text_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))
        self.video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))

        self.visual_token_selector = VisualTokenSelection(self.task_config.max_frames, embed_dim, topk=3)

        self.loss_fct = CrossEn()
        self.loss_mil = MILNCELoss_BoF()

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()

        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # [bs, 1, dim], [bs, num_words, dim], [bs, num_frames, dim]
        cls, text_feat, frame_feat, patch_feat = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask, 
                                                                video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            patch_feat = allgather(patch_feat, self.task_config)
            frame_feat = allgather(frame_feat, self.task_config)
            cls = allgather(cls, self.task_config)
            text_feat = allgather(text_feat, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            # torch.distributed.barrier()

        if self.training:
            loss = 0.
            logits, ret,  *_tmp = self.get_similarity_logits(cls, text_feat, frame_feat, patch_feat, attention_mask, video_mask, shaped=True, loose_type=self.loose_type)
            sim_matrix1 = logits['logits1']
            # sim_matrix2 = logits['logits2']
            # sim_matrix3 = logits['logits3']

            loss1 = ret['loss1']
            loss2 = ret['loss2']
            
            sim_loss = (self.loss_fct(sim_matrix1) + self.loss_fct(sim_matrix1.T)) / 2
                        #  + self.loss_fct(sim_matrix2) + self.loss_fct(sim_matrix2.T) ) / 4
            loss_set = {'sim_loss': sim_loss, 'loss1': loss1, 'loss2': loss2}
            loss += sim_loss + loss1 + loss2

            return loss, loss_set
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        cls, text_feat = self.clip.encode_text(input_ids, return_hidden=True)
        cls, text_feat = cls.float(), text_feat.float()
        cls, text_feat = cls.contiguous(), text_feat.contiguous()

        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1)) #(128, 32, 512)
        cls = cls.view(bs_pair, -1, cls.size(-1))

        return cls, text_feat

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        # print("video shape:", video.shape)

        frame_feat, patch_feat = self.clip.encode_image(video,return_hidden=True, video_frame=video_frame)
        frame_feat, patch_feat  = frame_feat.float(), patch_feat.float()
        frame_feat, patch_feat = frame_feat.contiguous(), patch_feat.contiguous()

        embed_dim = frame_feat.size(-1)

        patch_feat = patch_feat.view(bs_pair, -1, patch_feat.size(-1)) # shape here should be (bs, max_frames*sample_len, hid_dim)
        patch_feat = self.visual_token_selector(patch_feat)

        patch_feat = patch_feat.reshape(bs_pair, self.task_config.max_frames, -1, embed_dim)
        frame_feat = frame_feat.view(bs_pair, -1, frame_feat.size(-1))
        frame_feat = self.agg_video_feat(frame_feat, video_mask, )

        return frame_feat, patch_feat

    def agg_video_feat(self, video_feat, video_mask, sim_header=None):
        video_feat = video_feat.contiguous()
        # if sim_header == "None":
        #     pass
        # elif sim_header == "seqLSTM":
        #     # Sequential type: LSTM
        #     video_feat_original = video_feat
        #     video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
        #                                       batch_first=True, enforce_sorted=False)
        #     video_feat, _ = self.lstm_visual(video_feat)
        #     if self.training: self.lstm_visual.flatten_parameters()
        #     video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
        #     video_feat = torch.cat(
        #         (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
        #     video_feat = video_feat + video_feat_original
        # elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
        video_feat_original = video_feat
        seq_length = video_feat.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
        position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        video_feat = video_feat + frame_position_embeddings
        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
        video_feat = self.transformerClip(video_feat, extended_video_mask)
        video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
        video_feat = video_feat + video_feat_original

        return video_feat

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        cls, text_feat = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True) # [bs, 1, dim], [bs, num_words, dim]
        frame_feat, patch_feat = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)                  # [bs, num_frames, dim]

        return cls, text_feat, frame_feat, patch_feat

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, cls, text_feat, frame_feat, patch_feat, text_mask, video_mask, sim_header="meanP"):
        """
            sequence_output: CLS token of text       # [bs, 1, dim]
            seq_features: all tokens of text         # [bs, num_words, dim]
            visual_output: all frames of video       # [bs, num_frames, dim]
        """
        cls, frame_feat = cls.contiguous(), frame_feat.contiguous()
        text_feat, patch_feat = text_feat.contiguous(), patch_feat.contiguous()

        visual_output = frame_feat
        word_weights = self.text_feattype_weight(text_feat) #(128, 32, 28)
        text_feat = torch.einsum('bmd,bmn->bnd', text_feat, word_weights).contiguous() #（128,28, 512）
        # video_mask_f = video_mask[batch_index, mediods_ids].contiguous() 
        # expand_times = visual_pixel_output.shape[1] // video_mask.shape[1]
        # video_mask_p = video_mask_f.unsqueeze(1).repeat(1,1,expand_times).view(video_mask.shape[0], -1)

        # video-level visual feature 
        video_output_norm = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_output = self._mean_pooling_for_similarity_visual(video_output_norm, video_mask)
        video_output = video_output / video_output.norm(dim=-1, keepdim=True)                    # [bs, dim]

        ################### cluster
        batch_index = torch.arange(frame_feat.shape[0], dtype=torch.long, device=frame_feat.device).unsqueeze(-1)
        assign, mediods_ids = batch_fast_kmedoids_with_split(frame_feat, K=24)
        frame_feat = frame_feat[batch_index, mediods_ids].contiguous() # (B, K, D) 
        patch_feat = patch_feat[batch_index, mediods_ids].contiguous() # (B, K, P, D) 

        video_mask_f = video_mask[batch_index, mediods_ids].contiguous() 
        expand_times = patch_feat.shape[1] // video_mask.shape[1]
        video_mask_p = video_mask_f.unsqueeze(1).repeat(1,1,expand_times).view(video_mask.shape[0], -1)
        ################### cluster

        logit_scale = self.clip.logit_scale.exp()
        logits = {}
        ret = {}
        if self.training:
            B, C, D = text_feat.shape

            z_a_norm1 = (text_feat - text_feat.mean(-1, keepdim=True)) / text_feat.std(-1, keepdim=True)  # [bs1, n_tp, dim]
            z_b_norm1 = (frame_feat - frame_feat.mean(-1, keepdim=True)) / frame_feat.std(-1, keepdim=True)  # [bs2, n_fp, dim]
            c1 = torch.einsum('abd,acd->abc', z_a_norm1, z_b_norm1) / D# [bs, n_tp, n_fp]
            c1 = c1.sum(0) / B # [n_tp, n_fp]
            # loss
            on_diag1 = torch.diagonal(c1).add_(-1).pow_(2).sum() / C
            off_diag1 = c1.flatten()[1:].view(C - 1, C + 1)[:, :-1].pow_(2).sum() / C
            
            # feature-wise
            z_a_norm4 = text_feat
            z_a_norm4 = (z_a_norm4 - z_a_norm4.mean(0, keepdim=True)) / z_a_norm4.std(0, keepdim=True)  # [bs1, dim]
            z_b_norm4 = torch.sum(patch_feat, dim=2)
            z_b_norm4 = (z_b_norm4 - z_b_norm4.mean(0, keepdim=True)) / z_b_norm4.std(0, keepdim=True)  # [bs2, dim]
            c4 = torch.einsum('aub,aud->ubd', z_a_norm4, z_b_norm4).mean(0) / B # [dim, dim]
            # loss
            on_diag4 = torch.diagonal(c4).add_(-1).pow_(2).sum() / D
            off_diag4 = 0*c4.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
            

            instance_loss = 0.03*(on_diag1 + off_diag1)
            feature_loss = 0.035*(on_diag4)
        else:
            instance_loss = 0
            feature_loss = 0

        return logits, ret
    
    def get_base_logits(self, cls, video_pool, video_feat):
        cls = cls / cls.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
                # # video-sentence score 
        sentence_video_logits = torch.matmul(torch.matmul(cls, self.global_mat_weight), video_pool.t())
        # #  sentence-frame score 
        # sentence_frame_logits = torch.sum(torch.matmul(cls, video_feat.permute(0, 2, 1)) \
        # * torch.matmul(torch.softmax(torch.matmul(cls, video_feat.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()
        return (sentence_video_logits)
    
    def get_instance_logits(self, cls, text_feat, video_feat, video_mask):
        weight = torch.einsum('ad, bnd->abn', cls, video_feat) # [bs1, dim] [bs2, n_fp, dim] - [bs1, bs2, n_fp]
        weight = torch.softmax(weight/3, dim=-1)
        weight = torch.einsum('abn, bn->abn', weight, video_mask)
        video_feat = torch.einsum('abn, bnd->abd', weight, video_feat) # [bs2, n_fp, dim]

        cls_norm = cls / cls.norm(dim=-1, keepdim=True)
        video_feat_norm = video_feat / video_feat.norm(dim=-1, keepdim=True)
        text_feat_norm = text_feat / text_feat.norm(dim=-1, keepdim=True)
        retrieve_logits_instance = torch.einsum('aud, abd->abu', text_feat_norm, video_feat_norm) #[bs1, bs2, n_tp]
        retrieve_logits_instance = retrieve_logits_instance * torch.matmul(torch.softmax(retrieve_logits_instance / 1e-2, dim=-1), self.instance_mat_weight)
        retrieve_logits = torch.sum(retrieve_logits_instance, dim=-1)

        return retrieve_logits, retrieve_logits_instance
    
    def get_object_logits(self, cls, text_feat, patch_feat, refer_logits, video_mask):
        weight = torch.einsum('ad, bvkd->abvk', cls, patch_feat) # [bs1, dim] [bs2, n_fp, dim] - [bs1, bs2, n_fp]
        weight = torch.softmax(weight/3, dim=2)
        weight = torch.einsum('abvk, bv->abvk', weight, video_mask)
        patch_feat = torch.einsum('abvk, bvkd->abkd', weight, patch_feat) # [bs2, n_fp, dim]

        cls_norm = cls / cls.norm(dim=-1, keepdim=True)
        patch_feat_norm = patch_feat / patch_feat.norm(dim=-1, keepdim=True)
        text_feat_norm = text_feat / text_feat.norm(dim=-1, keepdim=True)
        retrieve_logits = torch.einsum('aud, abkd->abuk', text_feat_norm, patch_feat_norm) #[bs1, bs2, n_tp]
        retrieve_logits = torch.einsum('abu, abuk->abk', refer_logits, retrieve_logits)
        retrieve_logits = torch.sum(retrieve_logits * torch.matmul(torch.softmax(retrieve_logits / 1e-2, dim=-1), self.object_mat_weight), dim=-1)
        # retrieve_logits = (retrieve_logits + torch.einsum('ad, abkd->abk', cls_norm, patch_feat_norm).mean(-1)) / 2

        return retrieve_logits

    def weighted_token_wise_intersection(self, text_token, frame_token, attention_mask, video_mask):
        text_token = text_token / text_token.norm(dim=-1, keepdim=True)
        frame_token = frame_token / frame_token.norm(dim=-1, keepdim=True)

        text_weight = self.text_weight_fc(text_token).squeeze(2)  # B x N_t x D -> B x N_t
        text_weight.masked_fill_(torch.as_tensor((1 - attention_mask), dtype=torch.bool), float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        video_weight = self.video_weight_fc(frame_token).squeeze(2) # B x N_v x D -> B x N_v
        video_weight.masked_fill_(torch.as_tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
        video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v


        # token-wise interaction
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_token, frame_token])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, attention_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        text_sum = attention_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        return retrieve_logits

    def _attenion_over_fine_grained_sim_matrix(self, word_features, frame_features):
        bs_video, num_frames, dim_video = frame_features.shape
        bs_text, num_words, dim_text = word_features.shape

        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_mat_weight), frame_features.view(-1, dim_video).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]

        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, cls, text_feat, frame_feat, patch_feat, text_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits, ret = self._loose_similarity(cls, text_feat, frame_feat, patch_feat, text_mask, video_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(cls, frame_feat, text_mask, video_mask, )

        return retrieve_logits, ret, contrastive_direction
