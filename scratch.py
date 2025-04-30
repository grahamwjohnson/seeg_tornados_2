from utilities import manifold_utilities
from utilities import utils_functions
import numpy as np
import glob
import random
import pickle

# device = 1
# som_precomputed_path = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/bse_inference/train45/kohonen/64SecondWindow_32SecondStride_Reductionmean/all_pats/GPU0_ToroidalSOM_ObjectDict_smoothsec64_Stride32_subsampleFileFactor1_preictalSec3600_gridsize128_lr0.5with0.8223decay0.010000min_sigma102.4with0.7934decay1.0min_numfeatures1027140_dims1024_batchsize64_epochs20_rolled_v50_h10.pt'
# plot_data_path = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/bse_inference/train45/kohonen/64SecondWindow_32SecondStride_Reductionmean/all_pats/overlay_figure_object.pkl'
# som, checkpoint = manifold_utilities.load_kohonen(som_precomputed_path, device)
# undo_log = True # If U-matrix is log scaled
# smoothing_factor = 10 # Spline smoothing 

# save_dir = '/media/glommy1/tornados/tmp_plots'

# file_windowsecs = 1 
# file_stridesecs = 1
# rewin_windowsecs = 64
# rewin_strideseconds = 1
# dummy_dir = f'/media/glommy1/tornados/bse_inference/sheldrake_epoch1138_validation/latent_files/{file_windowsecs}SecondWindow_{file_stridesecs}SecondStride'
# dummy_files = glob.glob(f'{dummy_dir}/*.pkl')
# num_files = len(dummy_files)
# rand_idx = random.randint(0, num_files)
# rand_file = dummy_files[rand_idx]
# with open(rand_file, "rb") as f: latent_data_fromfile = pickle.load(f)
# context = latent_data_fromfile['windowed_weighted_means']
# ww_logvars = latent_data_fromfile['windowed_weighted_logvars']
# w_mogpreds = latent_data_fromfile['windowed_mogpreds']
# print(f"Original shape of context: {context.shape}")
# rewin_context, rewin_logvars, rewin_mogpreds = utils_functions.rewindow_data(
#     context, ww_logvars, w_mogpreds, file_windowsecs, file_stridesecs, rewin_windowsecs, rewin_strideseconds, reduction='mean')
# print(f"Rewindowed shape of rewin_context: {rewin_context.shape}")

# # create fake data
# context_length = 64
# predicted_length = 64
# total_plot_length = context_length + predicted_length
# context_start_idx = random.randint(0, rewin_context.shape[0] - total_plot_length)
# plot_context = rewin_context[context_start_idx:context_start_idx + context_length, :]
# ground_truth_future = rewin_context[context_start_idx + context_length - 1: context_start_idx + context_length + predicted_length, :] # Include overlap point for plotting purposes
# fake_predictions = np.concatenate([plot_context[-1:,:], np.random.rand(predicted_length,1024)], axis=0) # Include overlap point for plotting purposes
# pred_plot_axis = manifold_utilities.plot_kohonen_prediction(save_dir, som, plot_data_path, plot_context, ground_truth_future, fake_predictions, undo_log, smoothing_factor)  









# import yaml
# import torch
# from models.BSP import bsp_print_models_flow
# from utilities import utils_functions

# # Read in configuration file & setup the run
# config_f = 'config.yml'
# with open(config_f, "r") as f: kwargs = yaml.load(f,Loader=yaml.FullLoader)
# kwargs = utils_functions.exec_kwargs(kwargs) # Execute the arithmatic build into kwargs and reassign kwargs

# batchsize = 4
# fake_data = torch.rand(batchsize, 64, 2048)

# bsp_print_models_flow(x=fake_data, **kwargs)



import torch

torch.hub.set_dir('./.torch_hub_cache') # Set a local cache directory for testing
bse, som, bsp = torch.hub.load(
    'grahamwjohnson/seeg_tornados_2',
    'load',
    codename='sheldrake',
    pretrained=True,
    load_bse=True, 
    load_bsp=False,
    trust_repo='check',
    force_reload=True
)
print(bse)

# # rval_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/preprocessed_data/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/rapid_val'
# # rval_files = glob.glob(f"{rval_dir}/*.pkl")
# # f_idx = 0
# # with open(rval_files[f_idx], "rb") as f: data = pickle.load(f)


# # # Initialize & fill the data tensor 
# # batchsize = 8
# # padded_channels = 256
# # data_tensor_np = np.zeros([batchsize, self.transformer_seq_length, padded_channels, 1])

# # for bbb in range(batchsize):
# #     start_idx = np.random.randint(0, int(data.shape[1]/2))
# #     hash_channel_order = np.arange(0,data.shape[0])
# #     np.random.shuffle(hash_channel_order)

# #     end_idx = start_idx + self.encode_token_samples * self.transformer_seq_length
# #     tmp = np.zeros((padded_channels, self.transformer_seq_length * self.encode_token_samples), dtype=np.float16) # Create an empty data tensor with the correct shape
# #     for i, channel_idx in enumerate(hash_channel_order): # Fill the data tensor according to hash_channel_order
# #         if channel_idx != -1:  # Skip -1 (these positions will remain zero-padded)
# #             tmp[i, :] = data[channel_idx, start_idx:end_idx]  # Assign data for valid channels
# #     tmp = np.swapaxes(tmp.reshape(tmp.shape[0], self.transformer_seq_length, self.encode_token_samples), 0, 1)  # Reshape and swap axes
# #     data_tensor_np[bbb, :, :, :] = tmp

# # # Make it Torch tensor and give it a batch
# # x = torch.Tensor(data_tensor_np).to(x)


# # # ENCODE/DECODE
# # with torch.no_grad():
# #     z_pseudobatch, mean_pseudobatch, logvar_pseudobatch, mogpreds_pseudobatch, attW = self.gmvae(x, reverse=False) # No 1-shift if not causal masking
# #     z_token = z_pseudobatch.split(self.transformer_seq_length, dim=0)
# #     z_token = torch.stack(z_token, dim=0)
# #     x_hat = self.gmvae(z_token, reverse=True)  



# # SAMPLE PRIOR DECODE
# batchsize = 8
# z_token = torch.zeros([batchsize, self.transformer_seq_length, self.latent_dim]).to(x)
# for b in range(0,batchsize):
#     z_token[b, :, :], comp_probs = self.gmvae.module.prior.sample_prior(self.transformer_seq_length)
# x_hat = self.gmvae(z_token, reverse=True)  


# # PURE COMPONENT DECODE with directional stepping
# c = 0
# mu_comp = self.gmvae.module.prior.means[c,:]
# # Define a direction (e.g., random, or another learned vector)
# direction = torch.randn(1, self.latent_dim).to(x)  # shape [1, 1024]
# direction = direction / direction.norm()  # Optional: normalize to unit length
# step_size = 0.2
# # Generate steps: [0, 1, 2, ..., 511] for 512 steps
# steps = torch.arange(self.transformer_seq_length).float().unsqueeze(-1).to(x)  # shape [512, 1]
# # Linearly interpolate: mu_comp + (step_size * steps) * direction
# # Reshape mu_comp and direction for broadcasting
# sequence = mu_comp + (step_size * steps) * direction  # shape [512, 1024]
# # Repeat for batch dimension (8) if needed
# z_token = sequence.unsqueeze(0).expand(8, -1, -1)  # shape [8, 512, 1024]




# x_hat = self.gmvae(z_token_NOISE, reverse=True)  


# # Plotting
# import matplotlib.gridspec as gridspec
# import matplotlib.pylab as pl
# import seaborn as sns
# import pandas as pd

# savedir = self.model_dir + f"/TMP"
# epoch = self.epoch

# num_singlebatch_channels_recon = 1
# num_recon_samples = 256

# x_hat_CPU = x_hat.detach().cpu().numpy()
# x_CPU = x.detach().cpu().numpy()

# # Fuse the sequential decodes/predictions together
# x_fused = np.moveaxis(x_CPU, 3, 2)
# x_fused = x_fused.reshape(x_fused.shape[0], x_fused.shape[1] * x_fused.shape[2], x_fused.shape[3])
# x_fused = np.moveaxis(x_fused, 1, 2)

# x_hat_fused = np.moveaxis(x_hat_CPU, 3, 2)
# x_hat_fused = x_hat_fused.reshape(x_hat_fused.shape[0], x_hat_fused.shape[1] * x_hat_fused.shape[2], x_hat_fused.shape[3])
# x_hat_fused = np.moveaxis(x_hat_fused, 1, 2)

# batchsize = x_hat.shape[0]

# random_ch_idxs = [0]

# # Make new grid/fig
# if x_fused.shape[2] > num_recon_samples:
#     gs = gridspec.GridSpec(batchsize, num_singlebatch_channels_recon * 2) # *2 because beginning and end of transformer sequence
# else:
#     sqrt_num = int(np.ceil(np.sqrt(batchsize * num_singlebatch_channels_recon)))
#     gs = gridspec.GridSpec(sqrt_num, sqrt_num) 
#     subplot_iter = 0

# fig = pl.figure(figsize=(24, 24))
# palette = sns.cubehelix_palette(n_colors=2, start=3, rot=1) 
# for b in range(0, batchsize):
#     for c in range(0,len(random_ch_idxs)):
#         if x_fused.shape[2] > num_recon_samples: # If length of recon is bigger than desire visualized length, then plot only start and end of transformer tokens (may be overlap)
#             for seq in range(0,2):
#                 if seq == 0:
#                     x_decode_plot = x_fused[b, random_ch_idxs[c], :num_recon_samples]
#                     x_hat_plot = x_hat_fused[b, random_ch_idxs[c], :num_recon_samples]
#                     title_str = 'StartOfTransSeq'
#                 else:
#                     x_decode_plot = x_fused[b, random_ch_idxs[c], -num_recon_samples:]
#                     x_hat_plot = x_hat_fused[b, random_ch_idxs[c], -num_recon_samples:]   
#                     title_str = 'EndOfTransSeq'             

#                 df = pd.DataFrame({
#                     "Target": x_decode_plot,
#                     "Prediction": x_hat_plot
#                 })

#                 ax = fig.add_subplot(gs[b, c*2 + seq]) 
#                 sns.lineplot(data=df, palette=palette, linewidth=1.5, dashes=False, ax=ax)
#                 # ax.set_title(f"Ch:{random_ch_idxs[c]}\n{file_name[b]}, {title_str}", fontdict={'fontsize': 12, 'fontweight': 'medium'})

#                 pl.ylim(-1, 1) # Set y-axis limit -1 to 1

#         else: # Can fit entire seuqence into desired raw signal visualization length
#             x_decode_plot = x_fused[b, random_ch_idxs[c], :]
#             x_hat_plot = x_hat_fused[b, random_ch_idxs[c], :]

#             df = pd.DataFrame({
#                 "Target": x_decode_plot,
#                 "Prediction": x_hat_plot
#             })

#             row = int(subplot_iter/sqrt_num)
#             col = subplot_iter - (row * sqrt_num)
#             ax = fig.add_subplot(gs[row, col]) 
#             sns.lineplot(data=df, palette=palette, linewidth=1.5, dashes=False, ax=ax)
#             ax.set_title(f"{file_name[b]}", fontdict={'fontsize': 8, 'fontweight': 'medium'})

#             pl.ylim(-1, 1) # Set y-axis limit -1 to 1
#             subplot_iter = subplot_iter + 1
        
# fig.suptitle(f"Batches 0:{batchsize-1}, Ch:{random_ch_idxs}")
# if not os.path.exists(savedir): os.makedirs(savedir)
# savename_jpg = f"{savedir}/RealtimeRecon_epoch{epoch}_iter{iter_curr}_allbatch.jpg"
# pl.savefig(savename_jpg, dpi=200)
# pl.close(fig)   

# pl.close('all') 





















# loops = 32
# ch_idxs = np.arange(256)
# savedir = f"{self.model_dir}/TMP"

# #Get new channel order

# looped_mean_pseudobatch = torch.zeros([loops, x.shape[0], x.shape[1], kwargs['prior_mog_components'], self.latent_dim]).to(x)
# looped_logvar_pseudobatch = torch.zeros([loops, x.shape[0], x.shape[1], kwargs['prior_mog_components'], self.latent_dim]).to(x)
# looped_mogpreds_pseudobatch = torch.zeros([loops, x.shape[0], x.shape[1], kwargs['prior_mog_components']]).to(x)

# with torch.no_grad():
#     for i in range(loops):
#         np.random.shuffle(ch_idxs)
#         x_rand = x[:, :, ch_idxs, :]

#         # Embed the random order
#         _, m, l, pred, _ = self.gmvae(x_rand, reverse=False) 

#         mean = m.split(self.transformer_seq_length, dim=0)
#         mean = torch.stack(mean, dim=0)
#         logvar = l.split(self.transformer_seq_length, dim=0)
#         logvar = torch.stack(logvar, dim=0)
#         mogpreds = pred.split(self.transformer_seq_length, dim=0)
#         mogpreds = torch.stack(mogpreds, dim=0)

#         looped_mean_pseudobatch[i, :, :, :, :] = mean
#         looped_logvar_pseudobatch[i, :, :, :, :] = logvar
#         looped_mogpreds_pseudobatch[i, :, :, :] = mogpreds

# # Plot
# import matplotlib.gridspec as gridspec
# import matplotlib.pylab as pl
# import seaborn as sns
# import pandas as pd
# m_np = looped_mean_pseudobatch.cpu().detach().numpy()
# l_np = looped_logvar_pseudobatch.cpu().detach().numpy()
# pred_np = looped_mogpreds_pseudobatch.cpu().detach().numpy()

# gs = gridspec.GridSpec(1, 3) 
# fig = pl.figure(figsize=(20, 14))


# b = 1 # batch_idx
# c = 1 # component
# t = 0

# # Mean
# ax_curr = fig.add_subplot(gs[0, 0]) 
# plot_data = np.mean(m_np[:, b, :, c, :], axis=1) # mean across token
# sns.heatmap(
#     plot_data, 
#     cmap=sns.cubehelix_palette(as_cmap=True), 
#     ax=ax_curr)
# title = "Mean"
# ax_curr.set_title(title)

# # logvar
# ax_curr = fig.add_subplot(gs[0, 1]) 
# plot_data = np.mean(l_np[:, b, :, c, :], axis=1) # mean across token
# sns.heatmap(
#     plot_data, 
#     cmap=sns.cubehelix_palette(as_cmap=True), 
#     ax=ax_curr)
# title = "Logvar"
# ax_curr.set_title(title)

# # Mogpred
# ax_curr = fig.add_subplot(gs[0, 2]) 
# plot_data = np.mean(pred_np[:, b, :, :], axis=1) # mean across token
# sns.heatmap(
#     plot_data, 
#     cmap=sns.cubehelix_palette(as_cmap=True), 
#     ax=ax_curr)
# title = "All MogPred"
# ax_curr.set_title(title)


# os.makedirs(savedir, exist_ok=True)
# savename_jpg = f"{savedir}/{loops}loop_batch{b}_token{t}_component{c}.jpg"
# pl.savefig(savename_jpg, dpi=200)
# pl.close(fig)   

# pl.close('all')



































