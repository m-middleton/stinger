
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import mne

def eeg_ica_artifact_rejection(raw_voltage):
    filt_voltage = raw_voltage.filter(l_freq=1, h_freq=None)

    #raw_voltage.plot_psd()
    #raw_voltage.plot(duration=60, remove_dc=False)

    #ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_voltage)
    #ecg_epochs.plot_image(combine="mean")

    # EOG eye blinks
    eog_epochs = mne.preprocessing.create_eog_epochs(filt_voltage, baseline=(-0.5, -0.2))
    # eog_epochs.plot_image(combine="mean")
    # eog_epochs.average().plot_joint()

    # ICA to find signal components
    ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(filt_voltage)
    print(ica)

    # How much variance is explained by the components?
    explained_var_ratio = ica.get_explained_variance_ratio(filt_voltage)
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
        )

    # Plot the components and their properties
    #ica.plot_sources(raw_voltage, show_scrollbars=False)
    #ica.plot_components()

    # Plot overlay of raw data and cleaned data
    #ica.plot_overlay(raw_voltage, exclude=[0,3], picks="eeg")
    #ica.plot_properties(raw_voltage, picks=0)

    # Manually find components
    # ica.exclude = [0,3]  # indices chosen based on various plots above

    # # ica.apply() changes the Raw object in-place, so let's make a copy first:
    # reconst_raw = raw_voltage.copy()
    # ica.apply(reconst_raw)

    # raw_voltage.plot(show_scrollbars=False)
    # reconst_raw.plot(show_scrollbars=False)
    # del reconst_raw

    # Find Components using EOG electrodes automagically
    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw_voltage)
    ica.exclude = eog_indices
    print(f'ICA exclude: {ica.exclude}')

    # # barplot of ICA component "EOG match" scores
    # ica.plot_scores(eog_scores)

    # # plot diagnostics
    # ica.plot_properties(raw_voltage, picks=eog_indices)

    # # plot ICs applied to raw data, with EOG matches highlighted
    # ica.plot_sources(raw_voltage, show_scrollbars=False)

    # # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    # ica.plot_sources(eog_epochs)

    return ica

def process_eeg_epochs(raw_voltage, t_min, t_max, baseline_correction=[None,0]):#(None, 0)):
    events, single_events_dict = mne.events_from_annotations(raw_voltage)
    print(single_events_dict)
    print(f'events: {len(events)}')
    print(f'events in raw {events}')

    # reject bad data
    reject_criteria = dict(
        eeg=150e-6,      # unit: V (EEG channels)
        eog=250e-6      # unit: V (EOG channels)
    )

    flat_criteria = dict(eeg=1e-10) # 1 µV # 1e-6

    epochs = mne.Epochs(
        raw_voltage,
        events,
        event_id=single_events_dict,
        tmin=t_min,
        tmax=t_max,
        baseline=baseline_correction,
        event_repeated='drop',
        # reject=reject_criteria,
        flat=flat_criteria,
        preload=True,
        verbose=False,
    )

    #epochs.plot_drop_log()
    # epochs.plot()

    return epochs

def process_eeg_raw(raw_voltage, l_freq=None, h_freq=80, resample=200):
    print(raw_voltage)
    print(raw_voltage.info)

    #raw_voltage.plot(duration=60, proj=False, n_channels=len(raw_voltage.ch_names), remove_dc=False)
    #input("Press Enter to continue...")
    #asdas=asdasd

    raw_voltage = raw_voltage.filter(l_freq=l_freq, h_freq=h_freq)

    # raw_voltage.compute_psd(fmax=50).plot(picks="data", exclude="bads")
    # raw_voltage.plot(duration=5, n_channels=30)

    # # set up and fit the ICA
    # ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    # ica.fit(raw_voltage)
    # ica.exclude = [1, 2]  # details on how we picked these are omitted here
    # ica.plot_properties(raw_voltage, picks=ica.exclude)

    #events = np.delete(events, list(range(15,37)), 0)
    #print(f'events: {len(events)}')

    ica = eeg_ica_artifact_rejection(raw_voltage)
    ica.apply(raw_voltage)

    if resample is not None:
        raw_voltage = raw_voltage.resample(sfreq=resample)

    return raw_voltage