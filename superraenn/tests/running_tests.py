import matplotlib.pyplot as plt
from feature_extraction import read_in_LC_files, feat_peaks, feat_rise_and_decline, feat_slope, feat_int


def get_peak(mydir, fname):
    not_done = True
    with open(mydir+fname) as f:
        while not_done:
            line = f.readline()
            if 'PEAKMJD' in line:
                pmjd = float(line.split(':')[-1])
                not_done = False
    return(pmjd)


test_dir = '/Volumes/My Passport/ps1_sne/all_lcs/output_lc/'
filename = 'PS1_PS1MD_PSc330023.snana.dat'
my_message = read_in_LC_files([test_dir+filename], ['PSc330023'])

filt_dict = {'g': 0, 'r': 1, 'i': 2, 'z': 3}

for my_lc in my_message:
    my_lc.add_LC_info(zpt=27.5, mwebv=0.0, redshift=0.1820,
                      lim_mag=25.0, obj_type='SNIa')
    my_lc.get_abs_mags()
    my_lc.sort_lc()

    pmjd = get_peak(test_dir, filename)
    pmjd = my_lc.find_peak(pmjd)
    my_lc.shift_lc(pmjd)
    my_lc.cut_lc(100)
    my_lc.filter_names_to_numbers(filt_dict)
    my_gp, gp_mags = my_lc.make_dense_LC(4)

plt.plot(my_lc.times, my_lc.abs_mags)
plt.plot(my_lc.times, my_lc.dense_lc[:, 0, 0])
plt.plot(my_lc.times, my_lc.dense_lc[:, 1, 0])
plt.plot(my_lc.times, my_lc.dense_lc[:, 2, 0])
plt.plot(my_lc.times, my_lc.dense_lc[:, 3, 0])
plt.errorbar(my_lc.times, my_lc.abs_mags, yerr=my_lc.abs_mags_err, line_style=None)
plt.show()


feat_peaks([my_lc])
t_rises_all, t_falls_all = feat_rise_and_decline([my_lc], [my_gp], [gp_mags], 1, 4)
print(t_rises_all, t_falls_all)
slopes = feat_slope([my_lc], [my_gp], [gp_mags])
print(slopes)
ints = feat_int([my_lc], [my_gp], [gp_mags])
print(ints)
