import torch
import torch.nn as nn
import os
from termcolor import colored

class SGDiff(nn.Module):

    def __init__(self, type, diff_opt, vocab, replace_latent=False, with_changes=True,
                 residual=False, gconv_pooling='avg', with_angles=False, clip=True, use_image=True, separated=False):
        super().__init__()
        assert type in ['mmgdreamer'], '{} is not included'.format(type)

        self.type_ = type
        self.vocab = vocab
        self.with_angles = with_angles
        self.epoch = 0
        self.diff_opt = diff_opt
        assert replace_latent is not None and with_changes is not None
        if self.type_ == 'mmgdreamer':
            from model.mmgdreamer import Sg2ScDiffModel
            self.diff = Sg2ScDiffModel(vocab, self.diff_opt, diffusion_bs=16, embedding_dim=64, mlp_normalization="batch", separated=separated,
                              gconv_num_layers=5, use_angles=with_angles, replace_latent=replace_latent, residual=residual, use_clip=clip, use_image=use_image)
        else:
            raise NotImplementedError
        self.diff.optimizer_ini()
        self.counter = 0

    def forward_mani(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat, dec_objs, dec_objs_grained,
                     dec_triples, dec_boxes, dec_angles, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, dec_objs_to_scene, missing_nodes,
                     manipulated_nodes, epoch=None, mask_text_objs=None, mask_rel_triples=None):

        if self.type_ == 'mmgdreamer':
            obj_selected, shape_loss, layout_loss, loss_dict = self.diff.forward(
                enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat, dec_objs, dec_objs_grained, dec_triples, dec_boxes,
                encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, dec_objs_to_scene, missing_nodes,
                manipulated_nodes, dec_sdfs, dec_angles, epoch, mask_text_objs, mask_rel_triples)
        else:
            raise NotImplementedError

        return obj_selected, shape_loss, layout_loss, loss_dict

    def load_networks(self, exp, epoch, strict=True, restart_optim=False, load_shape_branch=True):
        ckpt = torch.load(os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch)))
        diff_state_dict = {}
        diff_state_dict['opt'] = ckpt.pop('opt')
        if load_shape_branch:
            try:
                diff_state_dict['vqvae'] = ckpt.pop('vqvae')
                diff_state_dict['shape_df'] = ckpt.pop('shape_df')
                self.diff.ShapeDiff.vqvae.load_state_dict(diff_state_dict['vqvae'])
                self.diff.ShapeDiff.df.load_state_dict(diff_state_dict['shape_df'])
                self.diff.ShapeDiff.df_module = self.diff.ShapeDiff.df
                self.diff.ShapeDiff.vqvae_module = self.diff.ShapeDiff.vqvae
                print(colored(
                    '[*] shape branch has successfully been restored from: %s' % os.path.join(exp, 'checkpoint',
                                                                                              'model{}.pth'.format(
                                                                                                  epoch)), 'blue'))
            except:
                print('no vqvae or shape_df recorded. Assume it is only the layout branch')
        try:
            self.epoch = ckpt.pop('epoch')
            self.counter = ckpt.pop('counter')
        except:
            print('no epoch or counter recorded.')

        ckpt.pop('vqvae', None)
        ckpt.pop('shape_df', None)
        self.diff.load_state_dict(ckpt, strict=strict) # layout branch only
        print(colored('[*] GCN and layout branch has successfully been restored from: %s' % os.path.join(exp, 'checkpoint',
                                                                                    'model{}.pth'.format(epoch)),
                      'blue'))

        if not restart_optim:
            import torch.optim as optim
            self.diff.optimizerFULL.load_state_dict(diff_state_dict['opt'])
            self.diff.scheduler = optim.lr_scheduler.LambdaLR(self.diff.optimizerFULL, lr_lambda=self.diff.lr_lambda,
                                                    last_epoch=int(self.counter - 1))


    def sample_box_and_shape(self, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, gen_shape=False):
        if self.type_ == 'mmgdreamer':
            shape_dict, layout_dict = self.diff.sample(dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, gen_shape=gen_shape)
            return {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError

    def sample_boxes_and_shape_with_changes(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                            dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, manipulated_nodes, gen_shape=False):
        if self.type_ == 'mmgdreamer':
            keep, shape_dict, layout_dict = self.diff.sample_with_changes(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                                                    dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, manipulated_nodes, gen_shape=gen_shape)
            return keep, {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError

    def sample_boxes_and_shape_with_additions(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                            dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, missing_nodes, gen_shape=False):
        if self.type_ == 'mmgdreamer':
            keep, shape_dict, layout_dict = self.diff.sample_with_additions(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                                                    dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, missing_nodes, gen_shape=gen_shape)
            return keep, {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError

    def sample_boxes_and_shape_with_changes_mask(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                            dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, manipulated_nodes, gen_shape=False, mask_text_objs_enc=None, mask_text_objs=None):
        if self.type_ == 'mmgdreamer':
            keep, shape_dict, layout_dict = self.diff.sample_with_changes_mask(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                                                    dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, manipulated_nodes, gen_shape=gen_shape, mask_text_objs_enc=mask_text_objs_enc, mask_text_objs=mask_text_objs)
            return keep, {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError
    
    def sample_boxes_and_shape_with_changes_mask_three(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                            dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, manipulated_nodes, gen_shape=False,mask_type=None):
        if self.type_ == 'mmgdreamer':
            keep, shape_dict, layout_dict = self.diff.sample_with_changes_mask_three(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                                                    dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, manipulated_nodes, gen_shape=gen_shape, mask_type=mask_type)
            return keep, {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError

    def sample_boxes_and_shape_with_additions_mask(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                            dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, missing_nodes, gen_shape=False, mask_text_objs_enc=None, mask_text_objs=None):
        if self.type_ == 'mmgdreamer':
            keep, shape_dict, layout_dict = self.diff.sample_with_additions_mask(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                                                    dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, missing_nodes, gen_shape=gen_shape, mask_text_objs_enc=mask_text_objs_enc, mask_text_objs=mask_text_objs)
            return keep, {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError
    def sample_boxes_and_shape_with_additions_mask_three(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                            dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, missing_nodes, gen_shape=False, mask_type=None):
        if self.type_ == 'mmgdreamer':
            keep, shape_dict, layout_dict = self.diff.sample_with_additions_mask_three(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_image_feat,
                                                                    dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, missing_nodes, gen_shape=gen_shape, mask_type=mask_type)
            return keep, {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError
    def sample_box_and_shape_mask(self, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, gen_shape=False, mask_type=None):
        if self.type_ == 'mmgdreamer':
            shape_dict, layout_dict = self.diff.sample_mask(dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, gen_shape=gen_shape, mask_type=mask_type)
            return {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError
    
    def sample_box_and_shape_mask_four(self, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, gen_shape=False, mask_text_objs=None, mask_rel_triples=None):
        if self.type_ == 'mmgdreamer':
            shape_dict, layout_dict = self.diff.sample_mask_four(dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, gen_shape=gen_shape, mask_text_objs=mask_text_objs, mask_rel_triples=mask_rel_triples)
            return {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError

    def save(self, exp, outf, epoch, counter=None):
        if self.type_ == 'mmgdreamer':
            torch.save(self.diff.state_dict(epoch, counter), os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
        else:
            raise NotImplementedError
