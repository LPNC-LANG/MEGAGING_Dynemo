import numpy as np
import glmtools as glm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
import mne
from osl_dynamics.analysis import power
import seaborn as sns
import pandas as pd

def fit_glm(
        input_data,
        group_assignments,
        covariates,
        dimension_labels=None,
        plot_verbose=False,
        save_path=""
    ):
    """Fit a General Linear Model (GLM) to an input data given a design matrix.

    Parameters
    ----------
    input_data : np.ndarray
        Data to fit. Shape must be (n_subjects, n_features1, n_features2, ...).
    subject_ids : list of str
        List of subject IDs. Should match the order of subjects in `input_data` 
        and `group_assignments`.
    group_assignments : np.ndarray
        1D numpy array containing group assignments. A value of 1 indicates Group 1 
        and a value of 2 indicates Group 2.
    dimension_labels : list of str
        Labels for the dimensions of an input data. Defaults to None, in which 
        case the labels will set as ["Subjects", "Features1", "Features2", ...].
    plot_verbose : bool
        Whether to plot the deisign matrix. Defaults to False.
    save_path : str
        File path to save the design matrix plot. Relevant only when plot_verbose 
        is set to True.
    
    Returns
    -------
    model : glmtools.fit.OLSModel
        A fiited GLM OLS model.
    design : glmtools.design.DesignConfig
        Design matrix object for GLM modelling.
    glm_data : glmtools.data.TrialGLMData
        Data object for GLM modelling.
    """
    if covariates is None:
        covariates = {}

    # Create GLM dataset
    glm_data = glm.data.TrialGLMData(
        data=input_data,
        **covariates,
        category_list=group_assignments,
        dim_labels=dimension_labels,
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Group1", rtype="Categorical", codes=1)
    DC.add_regressor(name="Group2", rtype="Categorical", codes=2)
    for name in covariates:
        DC.add_regressor(
            name=name,
            rtype="Parametric",
            datainfo=name,
            # preproc="demean",
        )
    DC.add_contrast( # C1
        name="GroupDiff",
        values=[-1, 1] + [0] * len(covariates),
    ) # contrast: Group 2 - Group 1
    DC.add_contrast( # C2
        name="OverallMean",
        values=[0.5, 0.5] + [0] * len(covariates),
    ) # contrast: (Group 1 + Group 2) / 2

    design = DC.design_from_datainfo(glm_data.info)
    if plot_verbose:
        design.plot_summary(show=False, savepath=save_path)

    # Fit GLM model
    model = glm.fit.OLSModel(design, glm_data)

    return model, design, glm_data

def max_stat_perm_test(
        glm_model,
        glm_data,
        design_matrix,
        pooled_dims,
        contrast_idx,
        n_perm=5000,
        metric="tstats",
        n_jobs=1,
        return_perm=False,
    ):
    """Perform a max-t permutation test to evaluate statistical significance 
       for the given contrast.

    Parameters
    ----------
    glm_model : glmtools.fit.OLSModel
        A fitted GLM OLS model.
    glm_data : glmtools.data.TrialGLMData
        Data object for GLM modelling.
    design_matrix : glmtools.design.DesignConfig
        Design matrix object for GLM modelling.
    pooled_dims : int or tuples
        Dimension(s) to pool over.
    contrast_idx : int
        Index indicating which contrast to use. Dependent on glm_model.
    n_perm : int, optional
        Number of iterations to permute. Defaults to 10,000.
    metric : str, optional
        Metric to use to build the null distribution. Can be 'tstats' or 'copes'.
    n_jobs : int, optional
        Number of processes to run in parallel.
    return_perm : bool, optional
        Whether to return a glmtools permutation object. Defaults to False.
    
    Returns
    -------
    pvalues : np.ndarray
        P-values for the features. Shape is (n_features1, n_features2, ...).
    perm : glm.permutations.MaxStatPermutation
        Permutation object in the `glmtools` package.
    """

    # Run permutations and get null distributions
    perm = glm.permutations.MaxStatPermutation(
        design_matrix,
        glm_data,
        contrast_idx=contrast_idx,
        nperms=n_perm,
        metric=metric,
        tail=0, # two-sided test
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    if metric == "tstats":
        print("Using tstats as metric")
        tstats = abs(glm_model.tstats[0])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        print("Using copes as metric")
        copes = abs(glm_model.copes[0])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    if return_perm:
        return pvalues, perm
    return pvalues


def cluster_perm_test(
        glm_model,
        glm_data,
        design_matrix,
        pooled_dims,
        contrast_idx,
        cft=3,
        n_perm=1000,
        metric="tstats",
        bonferroni_ntest=1,
        n_jobs=1,
        return_perm=False,
    ):
    """Perform a cluster permutation test to evaluate statistical significance 
       for the given contrast.

    Parameters
    ----------
    glm_model : glmtools.fit.OLSModel
        A fitted GLM OLS model.
    glm_data : glmtools.data.TrialGLMData
        Data object for GLM modelling.
    design_matrix : glmtools.design.DesignConfig
        Design matrix object for GLM modelling.
    pooled_dims : int or tuples
        Dimension(s) to pool over.
    contrast_idx : int
        Index indicating which contrast to use. Dependent on glm_model.
    cft : cluster forming threshold (# of adjacency points along the pooled dimension)
    n_perm : int
        Number of iterations to permute. Defaults to 1,000.
    metric : str, optional
        Metric to use to build the null distribution. Can be 'tstats' or 'copes'.
    bonferroni_ntest : int
        Number of tests to use for Bonferroni correction. Defaults to 1 (i.e., no
        Bonferroni correction applied).
    n_jobs : int, optional
        Number of processes to run in parallel.
    return_perm : bool, optional
        Whether to return a glmtools permutation object. Defaults to False.
    
    Returns
    -------
    obs : np.ndarray
        Statistic observed for all variables. Values can be 'tstats' or 'copes'
        depending on the `metric`. Shape is (n_freqs,).
    clusters : list of np.ndarray
        List of ndarray, each of which contains the indices that form the given 
        cluster along the tested dimension. If bonferroni_ntest was given, clusters 
        after Bonferroni correction are returned.
    perm : glm.permutations.ClusterPermutation
        Permutation object in the `glmtools` package.
    """

    # Get metric values and define cluster forming threshold
    if metric == "tstats":
        obs = np.squeeze(glm_model.tstats)
        cft = cft
    if metric == "copes":
        obs = np.squeeze(glm_model.copes)
        cft = 0.001

    # Run permutations and get null distributions
    perm = glm.permutations.ClusterPermutation(
        design=design_matrix,
        data=glm_data,
        contrast_idx=contrast_idx,
        nperms=n_perm,
        metric=metric,
        tail=0, # two-sided test
        cluster_forming_threshold=cft,
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )

    # Extract significant clusters
    print(f"Bonferroni correction for {bonferroni_ntest} modes ...")
    percentile = (1 - (0.05 / bonferroni_ntest)) * 100
    # NOTE: We use alpha threshold of 0.05.
    clu_masks, clu_stats = perm.get_sig_clusters(glm_data, percentile)
    if clu_stats is not None:
        n_clusters = len(clu_stats)
    else: n_clusters = 0
    print(f"After Bonferroni correction: Found {n_clusters} clusters")
    clusters = [
            np.arange(len(clu_masks))[clu_masks == n]
            for n in range(1, n_clusters + 1)
        ]

    if return_perm:
        return obs, clusters, perm
    return obs, clusters

def evoked_response_max_stat_perm(
    data, n_perm, covariates=None, metric="copes", n_jobs=1
):
    """Statistical significant testing for evoked responses.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a sign flip permutations test with the maximum statistic to
    determine a p-value for evoked responses.

    Parameters
    ----------
    data : np.ndarray
        Baseline corrected evoked responses. This will be the target data for
        the GLM. Must be shape (n_subjects, n_samples, ...).
    n_perm : int
        Number of permutations.
    covariates : dict, optional
        Covariates (extra regressors) to add to the GLM fit. These will be
        z-transformed. Must be of shape (n_subjects,).
    metric : str, optional
        Metric to use to build the null distribution. Can be :code:`'tstats'` or
        :code:`'copes'`.
    n_jobs : int, optional
        Number of processes to run in parallel.

    Returns
    -------
    pvalues : np.ndarray
        P-values for the evoked response. Shape is (n_subjects, n_samples, ...).
    """
    print("Covariates are not demeaned or normalized")
    
    if covariates is None:
        covariates = {}

    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")

    ndim = data.ndim
    if ndim < 3:
        raise ValueError("data must be 3D or greater.")

    # Create GLM Dataset
    data = glm.data.TrialGLMData(
        data=data,
        **covariates,
        dim_labels=["subjects", "time"] + [f"features {i}" for i in range(1, ndim - 1)],
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    for name in covariates:
        DC.add_regressor(
            name=name,
            rtype="Parametric",
            datainfo=name,
            # preproc="z",
        )
    DC.add_regressor(name="Mean", rtype="Constant")
    DC.add_contrast(name="Mean", values=[1] + [0] * len(covariates))
    design = DC.design_from_datainfo(data.info)

    # Fit model and get t-statistics
    model = glm.fit.OLSModel(design, data)

    # Pool over all dimensions over than subjects
    pooled_dims = tuple(range(1, ndim))

    # Run permutations and get null distribution
    perm = glm.permutations.MaxStatPermutation(
        design,
        data,
        contrast_idx=0,  # selects the Mean contrast
        nperms=n_perm,
        metric=metric,
        tail=0,  # two-sided test
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    if metric == "tstats":
        print("Using tstats as metric")
        tstats = abs(model.tstats[0])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        print("Using copes as metric")
        copes = abs(model.copes[0])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100
    
    return pvalues


###################################
# PLOTTING
###################################

def get_time_cluster_indices(
        time_data,
        group_assignments,
        n_perm,
        bonferroni_ntest,
        pooled_dims,
        covariates=None,
        cft=3,   
        contrast_idx=0,
    ):

    # Number of states (modes)
    n_states = time_data.shape[2]

    # Initialize list to store cluster indices for each state
    cluster_indices_per_state = []

    for n in range(n_states):
        # Fit GLM
        time_model, time_design, time_glm_data = fit_glm(
            input_data=time_data[:, :, n],
            group_assignments=group_assignments,
            covariates=covariates,
            dimension_labels=["Subjects", "Timepoints"]
        )

        # Perform cluster permutation tests on state-specific time courses
        _, clu_idx = cluster_perm_test(
            time_model,
            time_glm_data,
            time_design,
            pooled_dims=pooled_dims,
            cft=cft,
            contrast_idx=contrast_idx,
            metric="tstats",
            n_perm=n_perm,
            bonferroni_ntest=bonferroni_ntest,
        )
        # Store cluster indices for this state
        cluster_indices_per_state.append(clu_idx)
    
    return cluster_indices_per_state

def plot_evoked_response_with_clusters_indices(
    t,
    epochs,
    cluster_indices=None,
    offset_between_bars=0.02,
    labels=None,
    legend_loc=1,
    x_label=None,
    y_label=None,
    title=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Plot evoked responses with significant time points highlighted using cluster indices.

    Parameters
    ----------
    t : np.ndarray
        Time axis. Shape must be (n_samples,).
    epochs : np.ndarray
        Evoked responses. Shape must be (n_samples, n_channels).
    cluster_indices : list of np.ndarray
        List of arrays, each containing indices of significant time points for each channel.
    significance_level : float, optional
        Value to threshold the p-values with to consider significant (not used here).
    offset_between_bars : float, optional
        Vertical offset between bars that highlight significance.
    labels : list, optional
        Label for each evoked response time series.
    legend_loc : int, optional
        Position of the legend.
    x_label : str, optional
        Label for x-axis.
    y_label : str, optional
        Label for y-axis.
    title : str, optional
        Figure title.
    fig_kwargs : dict, optional
        Arguments to pass to :code:`plt.subplots()`.
    ax : plt.axes, optional
        Axis object to plot on.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`ax=None` and
        :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object(s). Only returned if :code:`ax=None` and
        :code:`filename=None`.
    """

    # Validation
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        else:
            if len(labels) != epochs.shape[1]:
                raise ValueError("Incorrect number of lines or labels passed.")
        add_legend = True
    else:
        labels = [None] * epochs.shape[1]
        add_legend = False

    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    if fig_kwargs is None:
        fig_kwargs = {}
    default_fig_kwargs = {"figsize": (7, 4)}
    fig_kwargs = {**default_fig_kwargs, **fig_kwargs}

    # Create figure
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(**fig_kwargs)

    for mode, e, l in zip(range(epochs.shape[1]),epochs.T,labels):
        # Plot evoked response
        p = ax.plot(t, e, label=l)

        # Highlight significant time points
        if cluster_indices is not None:
            c_indices = cluster_indices[mode]
            for cluster in range(len(c_indices)):
                sig_times = t[c_indices[cluster]]
                if len(sig_times) > 0:
                    y = 1.1 * np.max(epochs) + mode * offset_between_bars
                    dt = (t[1] - t[0]) / 2
                    for st in sig_times:
                        ax.plot(
                            (st - dt, st + dt),
                            (y, y),
                            color=p[0].get_color(),
                            linewidth=3,
                        )

    # Add a dashed line at time = 0
    ax.axvline(0, linestyle="--", color="black")

    # Set title, axis labels and range
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(t[0], t[-1])

    # Add a legend
    if add_legend:
        ax.legend(loc=legend_loc)

    # Save figure
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    elif create_fig:
        return fig, ax

def plot_evoked_response_with_clusters_indices_group_diff(
    t,
    epochs_group1,
    epochs_group2,
    cluster_indices=None,
    offset_between_bars=0.02,
    labels=None,
    legend_loc=1,
    x_label=None,
    y_label=None,
    title=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
    linestyle="dotted",
):
    # Validation
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        else:
            if len(labels) != epochs_group1.shape[1]:
                raise ValueError("Incorrect number of lines or labels passed.")
        add_legend = True
    else:
        labels = [None] * epochs_group1.shape[1]
        add_legend = False

    if ax is not None:
        if filename is not None:
            raise ValueError(
                "Please use plotting.save() to save the figure instead of the "
                + "filename argument."
            )
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    if fig_kwargs is None:
        fig_kwargs = {}
    default_fig_kwargs = {"figsize": (7, 4)}
    fig_kwargs = {**default_fig_kwargs, **fig_kwargs}

    # Create figure
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(**fig_kwargs)

    # Build a colormap
    qcmap = plt.rcParams["axes.prop_cycle"].by_key()["color"] # qualitative

    # Group1
    for mode, e, l in zip(range(epochs_group1.shape[1]),epochs_group1.T,labels):
        # Plot evoked response
        ax.plot(t, e, label=l, c=qcmap[mode])
        # Add a legend
        if add_legend:
            ax.legend(loc=legend_loc)
    #Group2
    for mode, e, l in zip(range(epochs_group2.shape[1]),epochs_group2.T,labels):
        # Plot evoked response
        ax.plot(t, e, label=l, c=qcmap[mode], linestyle=linestyle)

    for mode in range(epochs_group1.shape[1]):
        # Highlight significant time points
        if cluster_indices is not None:
            c_indices = cluster_indices[mode]
            for cluster in range(len(c_indices)):
                sig_times = t[c_indices[cluster]]
                if len(sig_times) > 0:
                    y = 1.1 * np.max(epochs_group1) + mode * offset_between_bars
                    dt = (t[1] - t[0]) / 2
                    for st in sig_times:
                        ax.plot(
                            (st - dt, st + dt),
                            (y, y),
                            color=qcmap[mode],
                            linewidth=3,
                        )

    # Add a dashed line at time = 0
    ax.axvline(0, linestyle="--", color="black")

    # Set title, axis labels and range
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(t[0], t[-1])

    # Save figure
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    elif create_fig:
        return fig, ax

    
def plot_spectra_with_clusters_indices(
        f,
        psd,
        group_assignments,
        n_perm,
        bonferroni_ntest,
        contrast_idx=0,
        cft=3,
        covariates=None,
        filename=None,
    ):
    """Plots state-specific PSDs and their between-group statistical differences.
    This function tests statistical differences using a cluster permutation test 
    on the frequency axis.
    """

    # Number of states
    n_states = psd.shape[1]

    # Get group-averaged PSDs
    gpsd_old = np.mean(psd[group_assignments == 2], axis=0)
    gpsd_young = np.mean(psd[group_assignments == 1], axis=0)
    # dim: (n_states, n_channels, n_freqs)

    # Build a colormap
    qcmap = plt.rcParams["axes.prop_cycle"].by_key()["color"] # qualitative

    # Plot state-specific PSDs and their statistical difference
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    k = 0 # subplot indices
    for n in range(n_states):
        print(f"Plotting State {n + 1}")
        
        # Set the row index
        if n != 0:
            k += 1
        
        # Fit GLM on state-specific, parcel-averaged PSDs
        ppsd = np.mean(psd, axis=2)
        # dim: (n_subjects, n_states, n_freqs)
        psd_model, psd_design, psd_data = fit_glm(
            input_data=ppsd[:, n, :],
            group_assignments=group_assignments,
            covariates=covariates,
            dimension_labels=["Subjects", "Frequency"],
        )

        # Perform cluster permutation tests on state-specific PSDs
        t_obs, clu_idx = cluster_perm_test(
            psd_model,
            psd_data,
            psd_design,
            pooled_dims=(1,),
            contrast_idx=contrast_idx,
            metric="tstats",
            cft=cft,
            n_perm=n_perm,
            bonferroni_ntest=bonferroni_ntest,
        )
        n_clusters = len(clu_idx)
        t_obs = t_obs[0] # select the first contrast

        # Average group-level PSDs over the parcels
        po = np.mean(gpsd_old[n], axis=0)
        py = np.mean(gpsd_young[n], axis=0)
        eo = np.std(gpsd_old[n], axis=0) / np.sqrt(gpsd_old.shape[0])
        ey = np.std(gpsd_young[n], axis=0) / np.sqrt(gpsd_young.shape[0])

        # Plot mode-specific group-level PSDs
        ax[k].plot(f, py, c=qcmap[n], label="Young")
        ax[k].plot(f, po, c=qcmap[n], label="Old", linestyle="--")
        ax[k].fill_between(f, py - ey, py + ey, color=qcmap[n], alpha=0.1)
        ax[k].fill_between(f, po - eo, po + eo, color=qcmap[n], alpha=0.1)
        if n_clusters > 0:
            for c in range(n_clusters):
                ax[k].axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)

        # Set labels
        ax[k].set_xlabel("Frequency (Hz)", fontsize=16)
        if k==0:
            ax[k].set_ylabel("PSD (a.u.)", fontsize=16)
        ax[k].set_title(f"Mode {n + 1}", fontsize=16)
        ax[k].axhline(0, linestyle="--", color="black")
        ax[k].ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
        ax[k].tick_params(labelsize=16)
        ax[k].yaxis.offsetText.set_fontsize(16)

        # Plot observed statistics
        end_pt = np.mean([py[int(len(py) // 3):], py[int(len(py) // 3):]])
        criteria = np.mean([ax[k].get_ylim()[0], ax[k].get_ylim()[1]])
        if end_pt >= criteria:
            inset_bbox = (0, -0.22, 1, 1)
        if end_pt < criteria:
            inset_bbox = (0, 0.28, 1, 1)
        ax_inset = inset_axes(ax[k], width='40%', height='30%', 
                            loc='center right', bbox_to_anchor=inset_bbox,
                            bbox_transform=ax[k].transAxes)
        ax_inset.plot(f, t_obs, color='k', lw=2) # plot t-spectra
        for c in range(len(clu_idx)):
            ax_inset.axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)
        ax_inset.set(
            xticks=np.arange(0, max(f), 20),
            ylabel="t-stats",
        )
        ax_inset.set_ylabel('t-stats', fontsize=14)
        ax_inset.tick_params(labelsize=14)

    plt.subplots_adjust(hspace=0.5)

    # Save figure
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    return None

def plot_spatial_with_clusters_indices(
        power_variance,
        AP_order, # A-P order of the channels
        group_assignments,
        n_perm,
        bonferroni_ntest,
        contrast_idx=0,
        cft=3,
        covariates=None,
    ):

    # Number of states
    mode = power_variance.shape[1]
    # A-P reordering for clustering
    power_variance_reordered=power_variance[:,:,AP_order]
    original_order= np.argsort(AP_order)

    # Plot state-specific PSDs and their statistical difference
    t_obs_list=[]
    for n in range(mode):
        print(f"Plotting Mode {n + 1}")

        # dim: (n_subjects, n_states, n_channels)
        power_model, power_design, power_data = fit_glm(
            input_data=power_variance_reordered[:, n, :],
            group_assignments=group_assignments,
            covariates=covariates,
            dimension_labels=["Subjects", "Channels"],
        )

        # Perform cluster permutation tests on state-specific PSDs
        t_obs, _ = cluster_perm_test(
            power_model,
            power_data,
            power_design,
            pooled_dims=(1,),
            contrast_idx=contrast_idx,
            metric="tstats",
            cft=cft,
            n_perm=n_perm,
            bonferroni_ntest=bonferroni_ntest,
        )
        t_obs_list.append(t_obs[0]) # select the first contrast

    for m in range(mode):
        power.save(
            t_obs_list[m][original_order], # Revert to the original ordering for plotting
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            subtract_mean=False,
            plot_kwargs={
                        "cmap": "RdBu_r",
                        "bg_on_data": True,
                        "darkness": 1,
                        "alpha": 1,
                        "vmin": -int(np.max(abs(t_obs_list[m]))),
                        "vmax": int(np.max(abs(t_obs_list[m]))),
                        "views": ['lateral'],
                    },
        )

    return None


def _categorise_pvalue(pval):
    """Assigns a label indicating statistical significance that corresponds 
    to an input p-value.

    Parameters
    ----------
    pval : float
        P-value from a statistical test.

    Returns
    -------
    p_label : str
        Label representing a statistical significance.
    """ 

    thresholds = [1e-3, 0.01, 0.05]
    labels = ["***", "**", "*", "n.s."]
    ordinal_idx = np.max(np.where(np.sort(thresholds + [pval]) == pval)[0])
    # NOTE: use maximum for the case in which a p-value and threshold are identical
    p_label = labels[ordinal_idx]

    return p_label

def plot_single_grouped_violin(
        data,
        group_label,
        filename,
        xlbl=None,
        ylbl=None,
        pval=None
    ):
    """Plots a grouped violin plot for each state.

    Parameters
    ----------
    data : np.ndarray
        Input data. Shape must be (n_subjects,).
    group_label : list of str
        List containing group labels for each subject.
    filename : str
        Path for saving the figure.
    xlbl : str
        X-axis tick label. Defaults to None. If you input a string of a number,
        it will print out "State {xlbl}".
    ylbl : str
        Y-axis tick label. Defaults to None.
    pval : np.ndarray
        P-values for each violin indicating staticial differences between
        the groups. If provided, statistical significance is plotted above the
        violins. Defaults to None.
    """

    # Validation
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data should be an numpy array.")

    # Build dataframe
    df = pd.DataFrame(data, columns=["Statistics"])
    df["Age"] = group_label
    df["State"] = np.ones((len(data),))

    # Plot grouped split violins
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.7, 4))
    sns.set_theme(style="white")
    vp = sns.violinplot(data=df, x="State", y="Statistics", hue="Age",
                        split=True, inner="box", linewidth=1,
                        palette={"Young": "b", "Old": "r"}, ax=ax)
    if pval is not None:
        vmin, vmax = [], []
        for collection in vp.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):
                vmin.append(np.min(collection.get_paths()[0].vertices[:, 1]))
                vmax.append(np.max(collection.get_paths()[0].vertices[:, 1]))
        vmin = np.min(np.array(vmin))
        vmax = np.max(np.array(vmax))
        ht = (vmax - vmin) * 0.045
        p_lbl = _categorise_pvalue(pval)
        if p_lbl != "n.s.":
            ax.text(
                vp.get_xticks(),
                vmax + ht,
                p_lbl, 
                ha="center", va="center", color="k", 
                fontsize=20, fontweight="bold"
            )
    sns.despine(fig=fig, ax=ax) # get rid of top and right axes
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] + np.max(vmax - vmin) * 0.05])
    if xlbl is not None:
        ax.set_xlabel(f"State {xlbl}", fontsize=22)
    else: ax.set_xlabel("")
    ax.set_ylabel(ylbl, fontsize=22)
    ax.set_xticks([])
    ax.tick_params(labelsize=22)
    ax.get_legend().remove()
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

    return None