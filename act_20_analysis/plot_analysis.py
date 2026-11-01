"""Render the saved audit arrays with NumPy and Matplotlib (no hardware needed)."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

OUT=Path(__file__).resolve().parent
D=np.load(OUT/'analysis_arrays.npz')
M=json.loads((OUT/'metrics.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':120,'savefig.dpi':180,'axes.grid':True,'grid.alpha':.18})
plots=[]


def save(fig,name,title,caption):
    fig.suptitle(title,fontsize=15)
    fig.tight_layout(rect=[0,0,1,.96])
    fig.savefig(OUT/(name+'.png'))
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)
    plots.append(dict(file=name+'.png',title=title,caption=caption))


def trajectories(key,refkey,name,unit,title):
    fig,axs=plt.subplots(4,2,figsize=(13,12))
    for j,ax in enumerate(axs.flat):
        if j==7:
            ax.axis('off');ax.text(.05,.9,'20 episodes overlaid\nSolid: measured state\nDashed: executed reference\nEach color is an episode\nTime includes lead-in and goal hold',va='top',transform=ax.transAxes);continue
        for ep in range(M['episodes']):
            mask=D['episode_id']==ep;t=D['t'][mask]
            ax.plot(t,D[key][mask,j],alpha=.55,lw=.8,color=plt.cm.tab20(ep))
            if refkey is not None:ax.plot(t,D[refkey][mask,j],alpha=.25,lw=.8,ls='--',color=plt.cm.tab20(ep))
        ax.set(title=f'Joint {j+1}',xlabel='Episode time (s)',ylabel=unit)
    save(fig,name,title,'Absolute joint coordinates; shared endpoints and smooth motion should form a narrow bundle. No episode was removed.')


trajectories('q','ref','01_joint_positions','Position (rad)','Joint positions: all 20 episodes and executed references')
trajectories('v',None,'02_joint_velocities','Velocity (rad/s)','Measured joint velocities: all 20 episodes')

fig,axs=plt.subplots(7,2,figsize=(13,17))
for j in range(7):
    for col,key,unit in [(0,'q','rad'),(1,'v','rad/s')]:
        ax=axs[j,col];x=D[key][:,j]
        ax.hist(x,bins=45,density=True,color='#3475a5' if col==0 else '#d58a35',alpha=.8)
        ax.axvline(np.median(x),color='black',ls='--',lw=1)
        ax.set(title=f'Joint {j+1}: {"position" if col==0 else "velocity"}',xlabel=unit,ylabel='Density')
save(fig,'03_joint_distributions','Joint position and velocity distributions',
     'Pooled observations, including lead-in and hold; dashed line is median. Peaks near zero velocity and endpoints are expected. Samples are temporally correlated, not independent trials.')

fig,axs=plt.subplots(7,2,figsize=(13,17))
for j in range(7):
    for col,key,unit in [(0,'phase_q','Position (rad)'),(1,'phase_v','Velocity (rad/s)')]:
        ax=axs[j,col];x=D[key][:,:,j]
        for ep,line in enumerate(x):ax.plot(D['phase'],line,color=plt.cm.tab20(ep),lw=.6,alpha=.35)
        mean=x.mean(axis=0);sd=x.std(axis=0)
        ax.fill_between(D['phase'],mean-sd,mean+sd,color='navy',alpha=.2)
        ax.plot(D['phase'],mean,color='navy',lw=1.4)
        ax.set(title=f'Joint {j+1}',xlabel='Motion phase time (s)',ylabel=unit)
save(fig,'04_phase_aligned_variation','Between-episode variation at the same motion phase',
     'Each motion is resampled to 121 points between A and B. Navy shows mean ± one population standard deviation across episodes; this is a descriptive spread, not a confidence interval.')

fig,axs=plt.subplots(1,2,figsize=(12,5))
for ax,key,title in zip(axs,['q','v'],['Position correlations','Velocity correlations']):
    im=ax.imshow(np.corrcoef(D[key],rowvar=False),vmin=-1,vmax=1,cmap='coolwarm')
    ax.set(xticks=range(7),yticks=range(7),xticklabels=range(1,8),yticklabels=range(1,8),xlabel='Joint',ylabel='Joint',title=title)
    for i in range(7):
        for j in range(7):ax.text(j,i,f'{np.corrcoef(D[key],rowvar=False)[i,j]:.2f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.025,pad=.03,label='Pearson correlation')
fig.subplots_adjust(right=.88,wspace=.35,top=.85)
# Colorbar layout is set explicitly rather than tightened.
fig.suptitle('Joint correlations across the collected observations',fontsize=15)
fig.savefig(OUT/'05_joint_correlations.png',bbox_inches='tight');fig.savefig(OUT/'05_joint_correlations.pdf',bbox_inches='tight');plt.close(fig)
plots.append(dict(file='05_joint_correlations.png',title='Joint correlations',caption='Correlations reflect the shared A-to-B task and added perturbations; they do not establish causal dependence.'))


def scree(ax,name):
    ratio=D[f'pca_{name}_explained_variance_ratio'];xs=np.arange(1,len(ratio)+1)
    ax.bar(xs,ratio*100,color='#3475a5',label='Individual PC')
    ax.plot(xs,np.cumsum(ratio)*100,'o-',color='#d17b28',label='Cumulative')
    ax.axhline(95,color='gray',ls=':',lw=1);ax.set(xlabel='Principal component',ylabel='Variance explained (%)',ylim=(0,105),xticks=xs)
    ax.legend(fontsize=8)


def loadings(ax,name,features):
    comp=D[f'pca_{name}_components'];xs=np.arange(len(features))
    ax.bar(xs-.18,comp[0],.36,label='PC1');ax.bar(xs+.18,comp[1],.36,label='PC2')
    ax.set(xticks=xs,xticklabels=features,ylabel='Component coefficient');ax.tick_params(axis='x',rotation=45);ax.legend()


fig,axs=plt.subplots(2,3,figsize=(16,9))
for row,name in enumerate(['position','velocity']):
    scree(axs[row,0],name);axs[row,0].set_title(name.title()+' explained variance')
    scores=D[f'pca_{name}_scores'];ax=axs[row,1]
    im=ax.scatter(scores[:,0],scores[:,1],c=D['clock'],s=5,cmap='viridis',alpha=.5,rasterized=True)
    ax.set(xlabel='PC1 score ('+('rad' if row==0 else 'rad/s')+')',ylabel='PC2 score',title=name.title()+' projection')
    fig.colorbar(im,ax=ax,label='Motion phase (s)')
    loadings(axs[row,2],name,[f'J{j}' for j in range(1,8)])
    axs[row,2].set_title(name.title()+' PC directions')
save(fig,'06_position_velocity_pca','PCA of measured joint positions and velocities',
     'Each 7D dataset is mean-centered and analyzed separately by SVD without standardizing joints. Position PCA uses rad; velocity PCA uses rad/s. Motion phase colors reveal the dominant task trajectory.')

fig,axs=plt.subplots(2,2,figsize=(13,10))
scree(axs[0,0],'combined');axs[0,0].set_title('14D standardized state: explained variance')
s=D['pca_combined_scores'];im=axs[0,1].scatter(s[:,0],s[:,1],c=D['clock'],s=6,cmap='viridis',alpha=.6,rasterized=True)
axs[0,1].set(xlabel='PC1 (standardized score)',ylabel='PC2',title='Combined state projection');fig.colorbar(im,ax=axs[0,1],label='Motion phase (s)')
features=[f'q{j}' for j in range(1,8)]+[f'dq{j}' for j in range(1,8)]
loadings(axs[1,0],'combined',features);axs[1,0].set_title('Position / velocity PC coefficients')
c=np.corrcoef(np.c_[D['q'],D['v']],rowvar=False);im=axs[1,1].imshow(c,vmin=-1,vmax=1,cmap='coolwarm')
axs[1,1].set(xticks=range(14),yticks=range(14),xticklabels=features,yticklabels=features,title='14D state correlation');axs[1,1].tick_params(axis='x',rotation=90);fig.colorbar(im,ax=axs[1,1],fraction=.045)
save(fig,'07_combined_state_pca','PCA of combined position and velocity state',
     'All 14 features are standardized individually to zero mean and unit population SD before SVD, preventing radians and radians/second from being mixed directly. Low-variance joints receive equal weight after standardization.')

fig,axs=plt.subplots(2,2,figsize=(13,10))
scree(axs[0,0],'motion_residual');axs[0,0].set_title('Motion residual explained variance')
s=D['pca_motion_residual_scores'];epids=np.repeat(np.arange(M['episodes']),len(D['phase']))
im=axs[0,1].scatter(s[:,0],s[:,1],c=epids,cmap='tab20',vmin=-.5,vmax=19.5,s=7,alpha=.6,rasterized=True)
axs[0,1].set(xlabel='Residual PC1 (rad)',ylabel='Residual PC2 (rad)',title='Residual projection by episode');fig.colorbar(im,ax=axs[0,1],label='Episode index')
loadings(axs[1,0],'motion_residual',[f'J{j}' for j in range(1,8)]);axs[1,0].set_title('Residual PC directions')
spread=D['phase_q'].std(axis=0)
for j in range(7):axs[1,1].plot(D['phase'],spread[:,j]*1000,label=f'J{j+1}')
axs[1,1].set(xlabel='Motion phase (s)',ylabel='Across-episode SD (mrad)',title='Measured position diversity at fixed phase');axs[1,1].legend(ncol=4,fontsize=8)
save(fig,'08_motion_residual_pca','PCA after subtracting the shared nominal A-to-B path',
     'Measured position minus analytical noise-free path, resampled to equal motion phases. These residuals include injected variation, controller bias, and tracking error. PCA is mean-centered, unscaled. This reveals variation hidden by the dominant common trajectory.')

eps=np.arange(M['episodes']);E=M['episode_metrics']
fig,axs=plt.subplots(2,3,figsize=(15,9))
axs[0,0].bar(eps,[e['max_tracking_error_rad']*1000 for e in E]);axs[0,0].set(ylabel='Maximum joint error (mrad)',title='Tracking error')
axs[0,1].bar(eps,[e['final_goal_error_rad']*1000 for e in E]);axs[0,1].axhline(20,color='orange',ls='--');axs[0,1].set(ylabel='Final max error (mrad)',title='Goal error; threshold 20 mrad')
axs[0,2].bar(eps,[e['max_noise_rad']*1000 for e in E]);axs[0,2].set(ylabel='Peak perturbation (mrad)',title='Action perturbation variation')
axs[1,0].plot(eps,[e['tick_dt_ms']['max'] for e in E],'o-',label='Maximum');axs[1,0].plot(eps,[e['tick_dt_ms']['median'] for e in E],'o-',label='Median');axs[1,0].set(ylabel='Observation period (ms)',title='15 Hz clock');axs[1,0].legend()
axs[1,1].plot(eps,[e['control_dt_ms']['max'] for e in E],'o-',label='Maximum');axs[1,1].plot(eps,[e['control_dt_ms']['median'] for e in E],'o-',label='Median');axs[1,1].set(ylabel='Reference update period (ms)',title='100 Hz reference clock');axs[1,1].legend()
axs[1,2].bar(eps,[e['last_tick_minus_last_command_ms'] for e in E]);axs[1,2].axhline(0,color='orange',ls='--');axs[1,2].set(ylabel='Last observation − last command (ms)',title='Recording ends before command stream stops')
for ax in axs.flat:ax.set_xlabel('Episode index');ax.set_xticks(range(0,20,2))
save(fig,'09_episode_quality','Motion and recording quality by episode',
     'Negative final timestamp differences confirm that recorded observations finish while the reference stream is still active. All seeds and all episodes are included.')

fig,axs=plt.subplots(3,2,figsize=(14,12));names=sorted(set(c['camera'] for c in M['camera_metrics']))
for col,name in enumerate(names):
    cs=[c for c in M['camera_metrics'] if c['camera']==name]
    axs[0,col].plot(eps,[c['mean_depth_valid_fraction']*100 for c in cs],'o-',label='Mean valid pixels')
    axs[0,col].plot(eps,[c['depth_valid_fraction']['min']*100 for c in cs],'o-',label='Worst frame')
    axs[0,col].set(title=name+' depth coverage',ylabel='Nonzero depth pixels (%)',ylim=(0,102));axs[0,col].legend()
    axs[1,col].plot(eps,[c['tick_age_ms']['median'] for c in cs],'o-',label='Median')
    axs[1,col].plot(eps,[c['tick_age_ms']['max'] for c in cs],'o-',label='Maximum')
    axs[1,col].set(title='Depth frame age at observation',ylabel='ms');axs[1,col].legend()
    axs[2,col].plot(eps,[c['rgb_tick_vs_indexed_full_mae_0_255']['median'] for c in cs],'o-',label='Median')
    axs[2,col].plot(eps,[c['rgb_tick_vs_indexed_full_mae_0_255']['max'] for c in cs],'o-',label='Maximum')
    axs[2,col].set(title='Tick RGB vs indexed full-rate RGB',ylabel='Mean pixel difference (0–255)');axs[2,col].legend()
for ax in axs.flat:ax.set_xlabel('Episode index');ax.set_xticks(range(0,20,2))
save(fig,'10_rgbd_quality','RGB-D coverage, timing, and stream association',
     'Every depth frame and every video frame was read. RGB association compares decoded 160×120 thumbnails; small differences are expected because the two videos are encoded independently. Nonzero depth coverage is a validity check, not proof of metric accuracy.')

fig,axs=plt.subplots(1,2,figsize=(13,5))
for name in names:
    hist=sum((D[f'depth_hist_{ep}_{name}'] for ep in range(M['episodes'])))
    centers=(np.arange(0,20000,100)+50)/1000
    binned=hist[:20000].reshape(200,100).sum(axis=1).astype(float);binned[0]-=hist[0]
    axs[0].plot(centers,binned/hist[1:].sum(),label=name)
    axs[1].plot(np.arange(1,20001)/1000,np.cumsum(hist[1:20001])/hist[1:].sum(),label=name)
axs[0].set(xlabel='Depth (m)',ylabel='Fraction of nonzero pixels / 0.1 m bin',title='Valid depth distribution',xlim=(0,10));axs[0].set_yscale('log');axs[0].legend()
axs[1].set(xlabel='Depth (m)',ylabel='Cumulative fraction of nonzero pixels',title='Valid depth cumulative distribution',xlim=(0,10));axs[1].legend()
save(fig,'11_depth_distributions','Depth distributions across every recorded pixel',
     'Depth is uint16 millimeters aligned to color. Zero pixels are excluded here and reported separately. The exocentric camera sees a farther background than the wrist camera; distant returns alone are not considered corruption.')

fig,axs=plt.subplots(1,2,figsize=(14,5));age=D['sensor_age'];fresh=D['sensor_fresh']
med=np.median(age,axis=0);p95=np.percentile(age,95,axis=0);xs=np.arange(age.shape[1])
axs[0].bar(xs,med,label='Median');axs[0].plot(xs,p95,'o',ms=3,label='95th percentile');axs[0].set(xlabel='Sensor index (CSV/metrics order)',ylabel='Packet age (ms)',title='Proximity packet age');axs[0].legend()
axs[1].bar(xs,fresh.mean(axis=0)*100);axs[1].set(xlabel='Sensor index',ylabel='Ticks with a new packet (%)',title='Sensor freshness at 15 Hz')
save(fig,'12_proximity_quality','Proximity telemetry quality',
     'Sensor values are latest-packet sample-and-hold. Sensor update rates are slower than the 15 Hz policy clock; repeated readings are expected. Camera and robot validity do not imply high-rate proximity sensing.')

summary={
 'episodes':M['episodes'],'observations':M['ticks'],'total_recorded_seconds':M['total_duration_s'],
 'issues':M['failures'], 'max_tracking_error_rad':max(e['max_tracking_error_rad'] for e in E),
 'max_goal_error_rad':max(e['final_goal_error_rad'] for e in E),
 'min_valid_depth_mean':{name:min(c['mean_depth_valid_fraction'] for c in M['camera_metrics'] if c['camera']==name) for name in names},
 'pca':M['pca']}
(OUT/'plot_summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'plot_index.json').write_text(json.dumps(plots,indent=2))
print(json.dumps(summary,indent=2))
