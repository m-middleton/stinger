
import numpy as np
from itertools import compress

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pywt

import mne

def nirs_ica_artifact_rejection(raw_intensity):
    filt_voltage = raw_intensity.filter(l_freq=1, h_freq=None)

    # ICA to find signal components
    ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(filt_voltage)
    print(ica)

    # Plot the components and their properties
    ica.plot_sources(raw_intensity, show_scrollbars=False)
    #ica.plot_components()

    # Manually find components
    ica.exclude = [0]  # indices chosen based on various plots above

    return ica

def process_nirs_epochs(raw_haemo, t_max, t_min, baseline_correction=None):
    events, single_events_dict = mne.events_from_annotations(raw_haemo)

    # Uncomment this for event processing
    # events, event_dict_true = mne.events_from_annotations(raw_haemo)
    # reversed_event_dict = {v: k for k, v in events_dict.items()}
    # single_events_dict = {reversed_event_dict[int(k)]:v for k, v in event_dict_true.items() if int(k) in reversed_event_dict.keys()}
    
    # print(f'events: {len(events)}')
    # print(events[np.isin(events[:,2],np.array([112,128,144]))])

    # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_haemo.info["sfreq"])
    # fig.subplots_adjust(right=0.7)  # make room for the legend
    # plt.show()

    # Epochs / reject / basline correction
    # reject_criteria = dict(hbo=80e-6)

    epochs = mne.Epochs(raw_haemo, 
                        events,
                        event_id=single_events_dict,
                        tmin=t_min, 
                        tmax=t_max,
                        # reject=reject_criteria, 
                        # reject_by_annotation=True,
                        proj=True, 
                        baseline=baseline_correction, 
                        event_repeated='drop',
                        preload=True,
                        detrend=None, 
                        verbose=True)
    #epoch_nirs.plot_drop_log()

    return epochs

def process_nirs_raw(raw_intensity, resample=10):
    print(raw_intensity)
    print(raw_intensity.info)

    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(
        raw_intensity.info, picks=picks
    )

    # Raw intensity
    # raw_intensity.pick(picks[dists > 0.01])
    # raw_intensity.plot(
    #     n_channels=len(raw_intensity.ch_names), duration=500, show_scrollbars=False
    # )

    # Raw optical
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    #raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)

    # Scalp coupling analysis
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    # fig, ax = plt.subplots(layout="constrained")
    # ax.hist(sci)
    # ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
    # plt.show()

    # Mark bad channels based on scalp coupling index
    #raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.75))
    #print(f'BADS: {raw_od.info["bads"]} SCI: {sci}\n\n\n\n\n\n\n')

    # Plot hemo
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    #raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=500, show_scrollbars=False)

    # Remove heart rate
    raw_haemo_unfiltered = raw_haemo.copy()
    # raw_haemo.filter(0.003, 0.8)#h_trans_bandwidth=0.2, l_trans_bandwidth=0.002)
    # raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)
    # for when, _raw in dict(Before=raw_haemo_unfiltered, After=raw_haemo).items():
    #     fig = _raw.compute_psd().plot(average=True, picks="data", exclude="bads")
    #     fig.suptitle(f"{when} filtering", weight="bold", size="x-large")
    # plt.show()

    # ICA to remove bold response
    #ica = nirs_ica_artifact_rejection(raw_haemo)
    #ica.apply(raw_haemo)

    # Wavlet Transform to remove physilogical interference
    raw_haemo.filter(0, 5)
    wavelet = 'sym6'

    coeffs = pywt.wavedec(raw_haemo.get_data(), wavelet, mode='symmetric', level=None)
    #coeffs = pywt.WaveletPacket(data=raw_haemo.get_data(), wavelet='db1', mode='symmetric', maxlevel=None)

    coeffs_to_use = coeffs.copy()
    for i in range(1,len(coeffs_to_use)-6):
        coeffs_to_use[-i] = np.zeros_like(coeffs_to_use[-i])

    filtered_data_dwt=pywt.waverec(coeffs_to_use,wavelet,mode='symmetric',axis=-1)

    fig, axs = plt.subplots(len(coeffs)+3, 1,figsize=(50,30))
    axs = list(axs)
    axs[0].plot(raw_haemo.get_data()[0])
    axs[0].set_title('intensity')

    raw_haemo._data = filtered_data_dwt
    axs[1].plot(raw_haemo.copy().filter(0.003, 0.08).get_data()[0])
    axs[1].set_title('band pass')

    raw_haemo._data = filtered_data_dwt
    axs[2].plot(raw_haemo.get_data()[0])
    axs[2].set_title('filtered')

    for id, signal in enumerate(coeffs):
        id = id+3
        axs[id].plot(signal[0])
        axs[id].set_title(f'{id} : {pywt.scale2frequency(wavelet, 2**id, precision=8)}')
    fig.tight_layout()
    pdf = PdfPages('nirs_processing.pdf')
    pdf.savefig()
    pdf.close()
    plt.close()
    #plt.show()

    # raw_haemo.plot()
    # input("Press Enter to continue...")

    # ica = nirs_ica_artifact_rejection(raw_haemo)
    # input("Press Enter to continue...")
    # asdas=asdasd

    if resample is not None:
        raw_haemo = raw_haemo.resample(sfreq=resample)

    return raw_haemo