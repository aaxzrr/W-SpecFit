import numpy as np 
import os
import pandas as pd
import obspy 
import sys
import matplotlib.pyplot as plt
import math
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import traceback

class Specfit: 
    """
    Specfit is a module for performing spectral fitting on waveforms 
    to determine t*, omega_0, and Fc.
    
    Output from the class is PDF and CSV
    """     
    def __init__(self, folder_output, folder_event, catalog, w1, w3=0, w5=0, report_pdf_csv=None,
                 start_fc=None, end_fc=None,
                 start_tstar=None, end_tstar=None,
                 start_SpectralLevel = None, end_SpectralLevel=None):
        self.f_output = folder_output
        self.f_event = folder_event
        self.catalog = catalog
        self.report_pdf_csv = report_pdf_csv
        self.w1 = w1
        self.w3 = w3
        self.w5 = w5
        self.start_fc = start_fc
        self.end_fc = end_fc
        self.start_tstar = start_tstar
        self.end_tstar = end_tstar
        self.start_speclvl = start_SpectralLevel
        self.end_speclvl = end_SpectralLevel
        
        # Initialize PDF
        if  report_pdf_csv is None:
            current_time = datetime.now().strftime("%H%M")
            self.report_pdf = f"report_{current_time}.pdf"
        else:
            self.report_pdf = f"{report_pdf_csv}.pdf"
        os.makedirs(folder_output, exist_ok=True)
        self.report_pdf = os.path.join(folder_output, self.report_pdf)
        self.pdf_canvas = canvas.Canvas(self.report_pdf, pagesize=A4)
        self.width, self.height = A4
        self.y_pos = self.height - 40
        self.plot_count = 0
        self.cols = 3  
        self.rows = 4 
        self.max_plots_per_page = self.cols * self.rows
        
        self.create_pdf_cover()
        
    @staticmethod
    def Cal_snr(signal, noise):
        """
        Calculate SNR
        """
        rmssignal=np.sqrt(np.mean(signal**2))
        rmsnoise=np.sqrt(np.mean(noise**2))
        snr = 10*np.log10((rmssignal/rmsnoise)**2)
        return snr
               
    @staticmethod
    def plot_log_v2(ax_log, x, y1, y2, y3, qc, name='figure_log'):
        """
        Plot logaritmic model, noise and signal
        """
        ax_log.loglog(x, y1, color='darkblue', label='Signal', lw=1)
        ax_log.loglog(x, y2, color='lime', label='noise', lw=1)
        ax_log.loglog(x, y3, color='red', label='model', lw=1)
        ax_log.set_xlabel('Frequency (Hz)')
        ax_log.set_ylabel('Amplitude')
        ax_log.legend(loc='upper right')
        if qc == 1:
            ax_log.set_title(name)
        else:
            ax_log.set_title(f'Fail_{name}')
            
    @staticmethod
    def plot_wave(ax_wave, mode, signal_noise_ms, w1, w2, w3, w4, w5, first, last):
        """
        Plot Wave and Pick.
        """
        # Define positions for plotting
        if mode == 'P':
            signal_noise_ms = signal_noise_ms[:(w4 + last) - (first - (w3 + w1 + w2))]
            xs = np.linspace(0, len(signal_noise_ms), len(signal_noise_ms)) / 100
            ax_wave.plot(xs, signal_noise_ms, 'k')
            noise_start = w2 / 100.0
            noise_end = (w2 + w5 + w1) / 100.0
            signal_start = (w2 + w5 + w3 + w1) / 100.0
            signal_end = (w2 + 2 * w5 + w3 + 2 * w1) / 100.0
        elif mode == 'S':
            xs = np.linspace(0, len(signal_noise_ms), len(signal_noise_ms)) / 100
            ax_wave.plot(xs, signal_noise_ms, 'k')
            noise_start = w2 / 100.0
            noise_end = (w2 + w5 + w1) / 100.0
            signal_start = first / 100.0
            signal_end = last / 100.0

        # Plot noise lines
        ax_wave.plot([noise_start, noise_start], [min(signal_noise_ms), max(signal_noise_ms)], color='lime', label='Noise')
        ax_wave.plot([noise_end, noise_end], [min(signal_noise_ms), max(signal_noise_ms)], color='lime')
        ax_wave.plot([noise_start, noise_end], [min(signal_noise_ms)] * 2, color='lime')

        # Plot signal lines
        ax_wave.plot([signal_start, signal_start], [min(signal_noise_ms), max(signal_noise_ms)], color='darkblue', label='Signal')
        ax_wave.plot([signal_end, signal_end], [min(signal_noise_ms), max(signal_noise_ms)], color='darkblue')
        ax_wave.plot([signal_start, signal_end], [min(signal_noise_ms)] * 2, color='darkblue')

        # Plot pick line
        ax_wave.plot([signal_start, signal_start], [min(signal_noise_ms), max(signal_noise_ms)], color='red', label='Pick')

        # Set labels and title
        ax_wave.set_xlabel('Time (second)')
        ax_wave.set_ylabel('Amplitude')
        ax_wave.legend(loc='upper right')
        ax_wave.set_title('Waveform')
        
            
    @staticmethod
    def update_progress(current, total):
        """
        A static method to display progress
        """
        percentage = (current / total) * 100
        bar = '#' * int(percentage // 2)
        dash = '-' * (50 - len(bar))
        sys.stdout.write(
            f"\rProgress: {percentage:.2f}% [{bar}{dash}] ({current}/{total}) "
        )
        sys.stdout.flush()

        
    def read_pha_file(self):
        """
        Parses the catalog file to extract event information
        """
        event = {}
        event_found = False
        with open(self.catalog, 'r') as file:
            lines = file.readlines()
        data = [line.split() for line in lines]
        for row in data:
            if len(row) > 0 and row[0] == '#':
                file_event = "{}{}{}_{}{}{}".format(row[1], row[2].zfill(2) if int(row[2])<10 else row[2],
                                                    row[3].zfill(2) if int(row[3])<10 else row[3], 
                                                    row[4].zfill(2) if int(row[4])<10 else row[4], 
                                                    row[5].zfill(2) if int(row[5])<10 else row[5], 
                                                    str(int(float(row[6]))).zfill(2) if int(float(row[6]))<10 else str(int(float(row[6]))))
                event_found=True
                event[file_event] = {}
                event[file_event].update({'num_event': int(row[-1])})
                event[file_event].update({'yymmdd': int(row[1] + row[2] + row[3])})
                event[file_event].update({'hh': int(row[4])})
                event[file_event].update({'mmss': int(row[5]) + (float(row[6]) / 60)})
                continue
            if event_found:
                if not row or row[0]=="#":
                    break
                else:
                    if 'STA' in event[file_event]:
                        event[file_event]['STA'].append(row)
                    else:
                        event[file_event].update({'STA':[row]})      
        return event
    
    def create_pdf_cover(self):
        """
        Create the cover page with a title.
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.pdf_canvas.setFont("Helvetica-Bold", 24)
        self.pdf_canvas.drawCentredString(self.width / 2, self.height - 200, "Processing Report")
        self.pdf_canvas.setFont("Helvetica", 16)
        self.pdf_canvas.drawCentredString(self.width / 2, self.height - 240, "Summary of Processed Files and QC Results")
        self.pdf_canvas.setFont("Helvetica", 12)
        self.pdf_canvas.drawCentredString(self.width / 2, self.height - 260, f"Created on: {current_time}")
        self.pdf_canvas.showPage()  
        

    def write_pdf(self, text, max_line_length=90):
        """
        Write text to PDF, handling page breaks and wrapping long lines, 
        ensuring each text starts on a new line.
        """
        self.y_pos -= 20  
        while len(text) > max_line_length:
            break_point = text[:max_line_length].rfind(' ')
            if break_point == -1:
                break_point = max_line_length
            self._print_line(text[:break_point])  
            text = text[break_point:].strip()  
        self._print_line(text)

    def _print_line(self, text):
        """
        Print a single line of text to the PDF and check if a new page is needed.
        """
        if self.y_pos < 60:
            self.pdf_canvas.showPage()
            self.y_pos = self.height - 40
        self.pdf_canvas.setFont("Helvetica", 12)
        self.pdf_canvas.drawString(100, self.y_pos, text)
        self.y_pos -= 10

    def save_pdf(self):
        """Save and close the PDF file."""
        self.pdf_canvas.save()

    def check_new_page(self):
        """Check if a new page is needed for plotting."""
        if self.plot_count >= self.max_plots_per_page:
            self.pdf_canvas.showPage()
            self.y_pos = self.height - 5
            self.plot_count = 0

    def chek_sta_data(self, station_data):
        """Looking for STA with full Pick (there are P and S)"""
        station_codes = [item[0] for item in station_data]
        duplicates = set([code for code in station_codes if station_codes.count(code) > 1])
        return list(duplicates)
            
    def main_process(self):
        """Main Process in spectral fitting method"""
        data_event = self.read_pha_file()
        total_files = len(data_event)
        if total_files == 0:
            print("Tidak ada file yang ditemukan.")
            return
        print(f"Processing {total_files} files...")
        self.update_progress(0, total_files)
        run_step = 2
        w2 = 20
        w4 = 200
        first = dict()
        last = dict()
        results = []
        event_succ=[]
        for current_index, event in enumerate(data_event,start=1):
            fc = []
            for step in range(run_step):
                try:
                    mean_fc = np.mean(fc) if fc else None
                    if f"{event}.mseed" in os.listdir(self.f_event):
                        st = obspy.read(os.path.join(self.f_event,f"{event}.mseed"))
                        station_data = data_event[event]['STA']
                        wave_p = {}
                        for tr in reversed(st):
                            sta = tr.stats.station
                            channel = tr.stats.channel
                            delta = tr.stats.delta
                            npts = tr.stats.npts
                            start_time = tr.stats.starttime
                            end_time = tr.stats.endtime
                            
                            start_hour_day = (start_time.hour*3600)+(start_time.day*24*3600)
                            
                            start_value = start_time.minute * 60 + start_time.second + (start_time.microsecond / 1_000_000)
                            end_value = ((start_time.day*24*3600) + (end_time.hour*3600) + (end_time.minute * 60) + end_time.second + 
                                         (end_time.microsecond / 1_000_000))-start_hour_day
                
                            cs_range = np.linspace(start_value, end_value, npts)
                            t_min = cs_range / 60
                            full_pick = self.chek_sta_data(station_data)
                            for _data in station_data:
                                station_match = tr.stats.station == _data[0].strip()
                                channel_match = tr.stats.channel in ["BHZ", "BHN"]
                                phase_match = (_data[3].strip() == 'P' if tr.stats.channel == "BHZ" else _data[3].strip() == 'S')                                                                 
                                
                                if station_match and channel_match and phase_match:
                                    data = np.array(tr.data) if not isinstance(tr.data, np.ndarray) else tr.data
                                    ta_real = data_event[event]['mmss']
                                    ta_sta = ta_real + (float(_data[1]) / 60)
                                                                
                                    name = f"{data_event[event]['num_event']}_{channel[-1]}_{data_event[event]['yymmdd']}_{data_event[event]['hh']}{round(data_event[event]['mmss'], 3)}_{sta}"
                                    first, last = None, None
                                    for i in range(len(cs_range) - 1):
                                        if (t_min[i]-1)%60+1 <= ta_sta and (t_min[i + 1]-1)%60+1 >= ta_sta:
                                            first = i - self.w5
                                            last = i + self.w1
                                            break             
                                    if first and last is not None:
                                        if channel =="BHZ":
                                            # Process signal and noise data
                                            signal_noise = data[first - (self.w3 + self.w1 + w2):]
                                            # Mean subtraction for centering signal to zero
                                            signal_noise_ms = signal_noise - np.mean(signal_noise)
                                            # save for S-pick
                                            wave_p[sta]=signal_noise_ms
                                            # Split into signal and noise components
                                            signal_ms = signal_noise_ms[(self.w3 + self.w5 + self.w1 + w2):(w2 + self.w5 + 2 * self.w1 + self.w3)]
                                            noise_ms = signal_noise_ms[w2:(w2 + self.w1)]
                                            # Perform spectral fitting and store results
                                            if step == 0:
                                                freq, spec_signal, spec_noise, Afix, result = self.spectral_fitting(signal_ms, noise_ms, delta)
                                                Spec_lvl, Fc, Tstar, error, snr, qc, resd = result
                                                if qc == 1:
                                                    fc.append(Fc)
                                                else:
                                                    continue 
                                            else:
                                                freq, spec_signal, spec_noise, Afix, result = self.spectral_fitting(signal_ms, noise_ms, delta, mean_fc)
                                                Spec_lvl, Fc, Tstar, error, snr, qc, resd = result
                                                results.append({
                                                    "name": name, "Spec_lvl": Spec_lvl, "Fc": Fc,
                                                    "Tstar": Tstar, "error": error, "snr": snr, "qc": qc, "residual":resd
                                                })
                                                # Plot waveforms and save images
                                                fig, (ax_log, ax_wave) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]})
                                                self.plot_log_v2(ax_log, freq, spec_signal, spec_noise, Afix, qc, name)
                                                self.plot_wave(ax_wave,'P', signal_noise_ms, self.w1, w2, self.w3, w4, self.w5, first, last)
                                        elif channel=="BHN" and sta in full_pick and sta in wave_p:
                                            signal_noise_ms = wave_p[sta]
                                            # Remove wavelength bias
                                            diff = len(t_min)-len(signal_noise_ms)
                                            first_n = first-diff
                                            last_n = last-diff
                                            # Split into signal and noise components
                                            signal_noise_ms = signal_noise_ms[:last_n+w4]
                                            signal_ms = signal_noise_ms[first_n:last_n]
                                            noise_ms = signal_noise_ms[w2:(w2 + self.w1)]
                                            # Perform spectral fitting and store results
                                            if step == 0:
                                                freq, spec_signal, spec_noise, Afix, result = self.spectral_fitting(signal_ms, noise_ms, delta)
                                                Spec_lvl, Fc, Tstar, error, snr, qc, resd = result
                                                if qc == 1:
                                                    fc.append(Fc)
                                                else:
                                                    continue                                         
                                            else:
                                                freq, spec_signal, spec_noise, Afix, result = self.spectral_fitting(signal_ms, noise_ms, delta, mean_fc)
                                                Spec_lvl, Fc, Tstar, error, snr, qc, resd = result
                                                results.append({
                                                    "name": name, "Spec_lvl": Spec_lvl, "Fc": Fc,
                                                    "Tstar": Tstar, "error": error, "snr": snr, "qc": qc, "residual":resd
                                                })
                                                # Plot waveforms and save images
                                                fig, (ax_log, ax_wave) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]})
                                                self.plot_log_v2(ax_log, freq, spec_signal, spec_noise, Afix, qc, name)
                                                self.plot_wave(ax_wave,'S', signal_noise_ms, self.w1, w2, self.w3, w4, self.w5, first_n, last_n)
                                        else:
                                            # Looking for time arival index
                                            for i in range(len(cs_range) - 1):
                                                if t_min[i] <= ta_real and t_min[i + 1] >= ta_real:
                                                    ta_index = i
                                            signal_noise = data[ta_index-self.w1-w2:]
                                            # Mean subtraction for centering signal to zero
                                            signal_noise_ms = signal_noise - np.mean(signal_noise)
                                                
                                            # Remove wavelength bias
                                            diff = len(t_min)-len(signal_noise_ms)
                                            first = first-diff
                                            last = last-diff
                                            signal_noise_ms = signal_noise_ms[:w4+last]
                                            # Split into signal and noise components
                                            signal_ms = signal_noise_ms[first:last]
                                            noise_ms = signal_noise_ms[w2:(w2 + self.w1)]
                                            # Perform spectral fitting and store results
                                            if step == 0:
                                                freq, spec_signal, spec_noise, Afix, result = self.spectral_fitting(signal_ms, noise_ms, delta)
                                                Spec_lvl, Fc, Tstar, error, snr, qc, resd = result
                                                if qc == 1:
                                                    fc.append(Fc)
                                                else:
                                                    continue
                                            else:
                                                freq, spec_signal, spec_noise, Afix, result = self.spectral_fitting(signal_ms, noise_ms, delta, mean_fc)
                                                Spec_lvl, Fc, Tstar, error, snr, qc, resd = result
                                                results.append({
                                                    "name": name, "Spec_lvl": Spec_lvl, "Fc": Fc,
                                                    "Tstar": Tstar, "error": error, "snr": snr, "qc": qc, "residual":resd
                                                })
                                                # Plot waveforms and save images
                                                fig, (ax_log, ax_wave) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]})
                                                self.plot_log_v2(ax_log, freq, spec_signal, spec_noise, Afix, qc, name)
                                                self.plot_wave(ax_wave,'S', signal_noise_ms, self.w1, w2, self.w3, w4, self.w5, first, last)
                                        if step == 1:
                                            plt.tight_layout()
                                            # Save figure based on qc value
                                            save_name =os.path.join(self.f_output, f'{name}.png')
                                            plt.savefig(save_name)
                                            plt.close(fig)  # More efficient than plt.clf()
                                            
                                            # Insert image into the PDF
                                            x_pos = 3 + (self.plot_count % self.cols) * (self.width // self.cols)
                                            self.pdf_canvas.drawImage(save_name, x_pos, self.y_pos - 190, width=190, height=190)
                                            # Increment plot count and check for new page
                                            self.plot_count += 1
                                            if self.plot_count % self.cols == 0:
                                                self.y_pos -= 190
                                                self.check_new_page()
                                        else:
                                            continue
                    event_succ.append(event)
                except Exception as e:
                    # Log error and continue processing
                    error_message = f"\nError processing {event}: {str(e)}"
                    self.write_pdf(f"\n{event}\nError: {error_message}")
                    traceback.print_exc()   
            if self.report_pdf_csv is None:
                csv_path = os.path.join(self.f_output, f'results.csv')
                pd.DataFrame(results).to_csv(csv_path, index=False)
            else:
                csv_path = os.path.join(self.f_output, f'{self.report_pdf_csv}.csv')
                pd.DataFrame(results).to_csv(csv_path, index=False)
            self.update_progress(current_index, total_files)
        self.save_pdf()

    def spectral_fitting(self, signal, noise, delta, fc = None):  
        """FFT on signal and noise""" 
        freq_nyq = 1/(2*delta)
        nfft = 2**int(math.ceil(math.log(len(signal),2)))
        sf = abs(np.fft.fft(signal,nfft))
        Spec_signal = sf[0:int(round(len(sf)/2))]/len(signal)
        freq = np.linspace(0,1,len(Spec_signal))*freq_nyq
        ns = abs(np.fft.fft(noise,nfft))
        Spec_noise = ns[0:int(round(len(sf)/2))]/len(noise)
        snr = self.Cal_snr(signal, noise)
        Afix, result = self.find_tstar(freq,Spec_signal,snr,nfft,fc)
        
        return freq, Spec_signal, Spec_noise, Afix, result
    
    def find_tstar(self, freq, filtered_spectrum, snr, nfft=None, mean_fc=None):
        """Grid Search for Tstar, Fc and spect_lvl(omega)"""
        if self.start_speclvl is not None and self.end_speclvl is not None:
            spec_lvl = np.linspace(self.start_speclvl, self.end_speclvl)
        else:
            rata_omega = np.mean(filtered_spectrum[0:6])
            spec_lvl = np.linspace(rata_omega/10, rata_omega*30)
        if self.start_fc is not None and self.end_fc is not None:
            fc = np.arange(self.start_fc, self.end_fc, 0.6)
        else:
            fc = np.arange(0.2,30,0.6)
        if self.start_tstar is not None and self.end_tstar is not None:
            tstar = np.arange(self.start_tstar, self.end_tstar, 0.001)
        else:
            tstar = np.arange(0.001, 0.50, 0.001)
        error = 10000
        
        rms    = dict()
        aa     = dict()
        
        # Grid search for Tstar, Fc and spect_lvl(omega)
        for i in range(len(fc) if mean_fc is None else 1):
            satu = (fc[i]**2) / (freq**2 + fc[i]**2) if mean_fc is None else (mean_fc**2) / (freq**2 + mean_fc**2)
            for j in range(0,len(spec_lvl)):
                dua = 2*np.pi*freq*spec_lvl[j]
                for k in range(0,len(tstar)):
                    tiga = np.exp(-np.pi*freq*tstar[k])
                    A = satu*dua*tiga 
                    if  nfft is not None:
                        Ac = A[1:int(nfft/2)]
                        residual = np.log(filtered_spectrum[1:int(nfft/2)])-np.log(Ac.conj().transpose())
                    else:
                        residual = np.log(filtered_spectrum)-np.log(A.conj().transpose())
                    rms[i,j,k] = np.sqrt(np.mean(residual**2))
                    resd = abs(residual[:int(len(residual)/3)])
                    aa [i, j, k] = np.mean(resd)
                    if rms[i, j, k] < error:
                        Fc = fc[i] if mean_fc is None else mean_fc
                        Spec_lvl = spec_lvl[j]
                        Tstar = tstar[k]
                        _Afix = A
                        error = rms [i, j, k]
                        resd_fix = aa[i, j, k] 

        if resd_fix < 0.8 and snr > 1.5:
            qc=1
        else:
            qc=0
        result = [Spec_lvl, Fc, Tstar, error, snr, qc, resd_fix]
        
        return _Afix, result
    
