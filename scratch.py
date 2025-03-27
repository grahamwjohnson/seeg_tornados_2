
loops = 32
ch_idxs = np.arange(256)
savedir = f"{self.model_dir}/TMP"

#Get new channel order

looped_mean_pseudobatch = torch.zeros([loops, x.shape[0], x.shape[1], kwargs['prior_mog_components'], self.latent_dim]).to(x)
looped_logvar_pseudobatch = torch.zeros([loops, x.shape[0], x.shape[1], kwargs['prior_mog_components'], self.latent_dim]).to(x)
looped_mogpreds_pseudobatch = torch.zeros([loops, x.shape[0], x.shape[1], kwargs['prior_mog_components']]).to(x)

with torch.no_grad():
    for i in range(loops):
        np.random.shuffle(ch_idxs)
        x_rand = x[:, :, ch_idxs, :]

        # Embed the random order
        _, m, l, pred, _ = self.gmvae(x_rand, reverse=False) 

        mean = m.split(self.transformer_seq_length, dim=0)
        mean = torch.stack(mean, dim=0)
        logvar = l.split(self.transformer_seq_length, dim=0)
        logvar = torch.stack(logvar, dim=0)
        mogpreds = pred.split(self.transformer_seq_length, dim=0)
        mogpreds = torch.stack(mogpreds, dim=0)

        looped_mean_pseudobatch[i, :, :, :, :] = mean
        looped_logvar_pseudobatch[i, :, :, :, :] = logvar
        looped_mogpreds_pseudobatch[i, :, :, :] = mogpreds

# Plot
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
import seaborn as sns
import pandas as pd
m_np = looped_mean_pseudobatch.cpu().detach().numpy()
l_np = looped_logvar_pseudobatch.cpu().detach().numpy()
pred_np = looped_mogpreds_pseudobatch.cpu().detach().numpy()

gs = gridspec.GridSpec(1, 3) 
fig = pl.figure(figsize=(20, 14))


b = 0 # batch_idx
c = 0 # component
t = 0

# Mean
ax_curr = fig.add_subplot(gs[0, 0]) 
plot_data = m_np[:, b, t, c, :]
sns.heatmap(
    plot_data, 
    cmap=sns.cubehelix_palette(as_cmap=True), 
    ax=ax_curr)
title = "Mean"
ax_curr.set_title(title)

# logvar
ax_curr = fig.add_subplot(gs[0, 1]) 
plot_data = l_np[:, b, t, c, :]
sns.heatmap(
    plot_data, 
    cmap=sns.cubehelix_palette(as_cmap=True), 
    ax=ax_curr)
title = "Logvar"
ax_curr.set_title(title)

# Mogpred
ax_curr = fig.add_subplot(gs[0, 2]) 
plot_data = pred_np[:, b, t, :]
sns.heatmap(
    plot_data, 
    cmap=sns.cubehelix_palette(as_cmap=True), 
    ax=ax_curr)
title = "All MogPred"
ax_curr.set_title(title)


os.makedirs(savedir, exist_ok=True)
savename_jpg = f"{savedir}/{loops}loop_batch{b}_token{t}_component{c}.jpg"
pl.savefig(savename_jpg, dpi=200)
pl.close(fig)   

pl.close('all')



































# HASH DECODING

# New Hash
modifier = -999
hash_pat_embedding = torch.zeros([8, 1024])
for i in range(len(file_name)):
    pat_id_curr = file_name[i].split("_")[0]
    num_channels_curr = np.sum(np.array(hash_channel_order[0]) != -1)
    hash_pat_embedding[i, :], _ = utils_functions.hash_to_vector(
        input_string = pat_id_curr,
        num_channels = num_channels_curr,
        padded_channels = 256,
        latent_dim = self.latent_dim,
        modifier = modifier, 
        hash_output_range = self.hash_output_range)


# Decode with new hash
x_hat = self.gmvae(z_token, reverse=True, hash_pat_embedding=hash_pat_embedding) 


# Plotting
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
import seaborn as sns
import pandas as pd

savedir = self.model_dir + f"/TMP"
epoch = self.epoch

num_singlebatch_channels_recon = 1
num_recon_samples = 256

x_hat_CPU = x_hat.detach().cpu().numpy()
x_CPU = x.detach().cpu().numpy()

# Fuse the sequential decodes/predictions together
x_fused = np.moveaxis(x_CPU, 3, 2)
x_fused = x_fused.reshape(x_fused.shape[0], x_fused.shape[1] * x_fused.shape[2], x_fused.shape[3])
x_fused = np.moveaxis(x_fused, 1, 2)

x_hat_fused = np.moveaxis(x_hat_CPU, 3, 2)
x_hat_fused = x_hat_fused.reshape(x_hat_fused.shape[0], x_hat_fused.shape[1] * x_hat_fused.shape[2], x_hat_fused.shape[3])
x_hat_fused = np.moveaxis(x_hat_fused, 1, 2)

batchsize = x_hat.shape[0]

random_ch_idxs = [0]

# Make new grid/fig
if x_fused.shape[2] > num_recon_samples:
    gs = gridspec.GridSpec(batchsize, num_singlebatch_channels_recon * 2) # *2 because beginning and end of transformer sequence
else:
    sqrt_num = int(np.ceil(np.sqrt(batchsize * num_singlebatch_channels_recon)))
    gs = gridspec.GridSpec(sqrt_num, sqrt_num) 
    subplot_iter = 0

fig = pl.figure(figsize=(24, 24))
palette = sns.cubehelix_palette(n_colors=2, start=3, rot=1) 
for b in range(0, batchsize):
    for c in range(0,len(random_ch_idxs)):
        if x_fused.shape[2] > num_recon_samples: # If length of recon is bigger than desire visualized length, then plot only start and end of transformer tokens (may be overlap)
            for seq in range(0,2):
                if seq == 0:
                    x_decode_plot = x_fused[b, random_ch_idxs[c], :num_recon_samples]
                    x_hat_plot = x_hat_fused[b, random_ch_idxs[c], :num_recon_samples]
                    title_str = 'StartOfTransSeq'
                else:
                    x_decode_plot = x_fused[b, random_ch_idxs[c], -num_recon_samples:]
                    x_hat_plot = x_hat_fused[b, random_ch_idxs[c], -num_recon_samples:]   
                    title_str = 'EndOfTransSeq'             

                df = pd.DataFrame({
                    "Target": x_decode_plot,
                    "Prediction": x_hat_plot
                })

                ax = fig.add_subplot(gs[b, c*2 + seq]) 
                sns.lineplot(data=df, palette=palette, linewidth=1.5, dashes=False, ax=ax)
                ax.set_title(f"Ch:{random_ch_idxs[c]}\n{file_name[b]}, {title_str}", fontdict={'fontsize': 12, 'fontweight': 'medium'})

                pl.ylim(-1, 1) # Set y-axis limit -1 to 1

        else: # Can fit entire seuqence into desired raw signal visualization length
            x_decode_plot = x_fused[b, random_ch_idxs[c], :]
            x_hat_plot = x_hat_fused[b, random_ch_idxs[c], :]

            df = pd.DataFrame({
                "Target": x_decode_plot,
                "Prediction": x_hat_plot
            })

            row = int(subplot_iter/sqrt_num)
            col = subplot_iter - (row * sqrt_num)
            ax = fig.add_subplot(gs[row, col]) 
            sns.lineplot(data=df, palette=palette, linewidth=1.5, dashes=False, ax=ax)
            ax.set_title(f"{file_name[b]}", fontdict={'fontsize': 8, 'fontweight': 'medium'})

            pl.ylim(-1, 1) # Set y-axis limit -1 to 1
            subplot_iter = subplot_iter + 1
        
fig.suptitle(f"Batches 0:{batchsize-1}, Ch:{random_ch_idxs}")
if not os.path.exists(savedir): os.makedirs(savedir)
savename_jpg = f"{savedir}/RealtimeRecon_epoch{epoch}_iter{iter_curr}_allbatch_MODIFIER{modifier}.jpg"
pl.savefig(savename_jpg, dpi=200)
pl.close(fig)   

pl.close('all') 