import numpy as np
import obspy
import obspy.taup
from matplotlib.figure import Figure


class Pssac(object):
    def __init__(self, canvas, config=None):
        self.canvas = canvas
        self.config = config
        self.baseurl = "./data/"

        self._handleConfig()

    def test(self):
        # TODO: delete this test
        x = range(0, 10)
        y = range(0, 200, 20)
        self.canvas.fig.clf()
        self.canvas.ax = self.canvas.fig.add_subplot(111)
        self.canvas.ax.plot(x, y)
        self.canvas.draw()

    def show(self):
        if(self.mode == "Event"):
            self._plotPreparation_eventMode()
        elif(self.mode == "Station"):
            self._plotPreparation_stationMode()

        self.canvas.fig.clf()
        if(self.aligned_phase == "s/S"):
            self._plot_s()
        elif(self.aligned_phase == "p/P"):
            self._plot_p()

        self._addText()
        self.canvas.draw()

    def _handleConfig(self):
        # TODO: Delete the ugly self.p=p, use __dict__.update()
        self.preset = self.config["preset"]
        self.afterset = self.config["afterset"]
        self.model = self.config["model"]
        self.scale = self.config["scale"]
        self.texted = self.config["texted"]
        self.y_range = self.config["y_range"]
        self.global_normal = self.config["global_normal"]
        self.aligned_phase = self.config["aligned_phase"]
        self.y_axis_type = self.config["y_axis_type"]
        self.filter_band = self.config["filter_band"]
        self.direction = self.config["direction"]
        self.mode = self.config["mode"]
        self.eventid = self.config["eventid"]
        self.UTCDateTime = self.config["UTCDateTime"]
        self.stationid = self.config["stationid"]
        self.station_selected = self.config["station_selected"]
        self.event_selected = self.config["event_selected"]
        # self.__dict__.update(self.config)

    def _addText(self):
        textX = np.zeros(len(self.stationid))
        textX[:] = self.afterset+10000/(self.afterset+self.preset)
        textY = self.shift_values
        if(self.texted == "None"):
            return
        elif(self.texted == "Event ID"):
            for i in range(len(self.stationid)):
                self.ax.text(textX[i], textY[i], self.stationid[i], fontsize=8)
        elif(self.texted == "Depth"):
            for i in range(len(self.stationid)):
                self.ax.text(textX[i], textY[i], self.depths[i], fontsize=8)
        elif(self.texted == "Azimuth"):
            for i in range(len(self.stationid)):
                self.ax.text(textX[i], textY[i],
                             f"{self.azimuths[i]:.2f}", fontsize=8)
        elif(self.texted == "Epicenter Distance"):
            for i in range(len(self.stationid)):
                self.ax.text(textX[i], textY[i],
                             f"{self.distance_degrees[i]:.2f}", fontsize=8)
        elif(self.texted == "Euclidean Distance"):
            for i in range(len(self.stationid)):
                self.ax.text(
                    textX[i], textY[i], f"{self.appro_line_distances[i]:.2f}", fontsize=8)

    def _calarrivals(self, st):
        model = obspy.taup.TauPyModel(model=self.model)
        gps = obspy.geodetics.base.gps2dist_azimuth(
            st.stats.sac["stla"], st.stats.sac["stlo"], st.stats.sac["evla"], st.stats.sac["evlo"])

        distance_km = gps[0]/1000
        azimuth = gps[2]
        antiAzimuth = gps[1]
        distance_degree = obspy.geodetics.base.kilometer2degrees(distance_km)
        depth = st.stats.sac["evdp"]

        parrival = model.get_travel_times(
            source_depth_in_km=st.stats.sac.evdp, distance_in_degree=distance_degree, phase_list=["p", "P"])[0].time
        sarrival = model.get_travel_times(
            source_depth_in_km=st.stats.sac.evdp, distance_in_degree=distance_degree, phase_list=["s", "S"])[0].time

        appro_line_distance = np.sqrt(distance_km**2+st.stats.sac.evdp**2)
        return distance_km, distance_degree, azimuth, antiAzimuth, parrival, sarrival, appro_line_distance, depth

    def _calsnr(self, data, starttime, parrival, sarrival):
        st = data.copy()
        noise = st.slice(starttime, parrival)
        signal = st.slice(parrival, sarrival)

        dn = noise.data
        ds = signal.data
        nptsn = noise.stats.npts
        nptss = signal.stats.npts

        pn = np.sum(dn**2)/nptsn
        ps = np.sum(ds**2)/nptss

        snr = 10*np.log10((ps-pn)/pn)
        return snr

    def _snr_color(self, x):
        if(x < 0):
            return 0.9
        elif(x > 30):
            return 0
        elif(np.isnan(x)):
            return 0.95
        else:
            return -0.03*x+0.9

    def _plotPreparation_eventMode(self):
        baseurl = self.baseurl+self.event_selected
        self.r = obspy.Stream()
        self.t = obspy.Stream()
        self.z = obspy.Stream()
        for station in self.stationid:
            self.r += obspy.read(baseurl+"/*"+station+"*R")
            self.t += obspy.read(baseurl+"/*"+station+"*T")
            self.z += obspy.read(baseurl+"/*"+station+"*Z")

        assert len(self.r) == len(self.t)
        assert len(self.r) == len(self.z)

        position = list(self.eventid).index(self.event_selected)
        self.time = np.zeros(len(self.r), dtype=np.object)
        self.time[:] = self.UTCDateTime[position]

        self.st = np.nan
        if(self.direction == "Vertical"):
            self.st = self.z.copy()
        elif(self.direction == "Radial"):
            self.st = self.r.copy()
        elif(self.direction == "Tangential"):
            self.st = self.t.copy()

        self.st = self.st.normalize(global_max=self.global_normal)

        self.st.filter("bandpass", freqmin=1. /
                       self.filter_band[1], freqmax=1./self.filter_band[0])

        self.distance_kms = np.zeros(len(self.r), dtype=np.float)
        self.distance_degrees = np.zeros(len(self.r), dtype=np.float)
        self.azimuths = np.zeros(len(self.r), dtype=np.float)
        self.antiAzimuths = np.zeros(len(self.r), dtype=np.float)
        self.parrivals = np.zeros(len(self.r), dtype=np.float)
        self.sarrivals = np.zeros(len(self.r), dtype=np.float)
        self.appro_line_distances = np.zeros(len(self.r), dtype=np.float)
        self.depths = np.zeros(len(self.r), dtype=np.float)

        for i in range(len(self.r)):
            self.distance_kms[i], self.distance_degrees[i], self.azimuths[i], self.antiAzimuths[
                i], self.parrivals[i], self.sarrivals[i], self.appro_line_distances[i], self.depths[i] = self._calarrivals(self.st[i])

        self.shift_values = np.zeros(len(self.r), dtype=np.float)
        if(self.y_axis_type == "Epicenter Distance"):
            self.shift_values = self.distance_degrees
        elif(self.y_axis_type == "Euclidean Distance"):
            self.shift_values = self.appro_line_distances
        elif(self.y_axis_type == "Depth"):
            self.shift_values = self.depths
        elif(self.y_axis_type == "Azimuth"):
            self.shift_values = self.azimuths

        if(np.max(self.shift_values) < self.y_range[1]):
            self.scale = (np.max(self.shift_values) -
                          np.min(self.shift_values))/15*self.scale
        else:
            self.scale = (self.y_range[1]-self.y_range[0])/15*self.scale

    def _plotPreparation_stationMode(self):
        self.r = obspy.Stream()
        self.t = obspy.Stream()
        self.z = obspy.Stream()
        self.time = np.empty(0, dtype=np.object)
        self.stationid_temp = np.empty(0, dtype="<U10")
        for i, event in enumerate(self.eventid):
            try:
                self.r += obspy.read(self.baseurl+event +
                                     "/*"+self.station_selected+"*R")
                self.t += obspy.read(self.baseurl+event +
                                     "/*"+self.station_selected+"*T")
                self.z += obspy.read(self.baseurl+event +
                                     "/*"+self.station_selected+"*Z")
                self.time = np.append(self.time, [self.UTCDateTime[i]])
                self.stationid_temp = np.append(
                    self.stationid_temp, [self.stationid[i]])
            except:
                pass

        self.UTCDateTime = self.time
        self.stationid = self.stationid_temp

        assert len(self.r) == len(self.t)
        assert len(self.r) == len(self.z)

        self.st = np.nan
        if(self.direction == "Vertical"):
            self.st = self.z.copy()
        elif(self.direction == "Radial"):
            self.st = self.r.copy()
        elif(self.direction == "Tangential"):
            self.st = self.t.copy()

        self.st = self.st.normalize(global_max=self.global_normal)

        self.st.filter("bandpass", freqmin=1. /
                       self.filter_band[1], freqmax=1./self.filter_band[0])

        self.distance_kms = np.zeros(len(self.r), dtype=np.float)
        self.distance_degrees = np.zeros(len(self.r), dtype=np.float)
        self.azimuths = np.zeros(len(self.r), dtype=np.float)
        self.antiAzimuths = np.zeros(len(self.r), dtype=np.float)
        self.parrivals = np.zeros(len(self.r), dtype=np.float)
        self.sarrivals = np.zeros(len(self.r), dtype=np.float)
        self.appro_line_distances = np.zeros(len(self.r), dtype=np.float)
        self.depths = np.zeros(len(self.r), dtype=np.float)

        for i in range(len(self.r)):
            self.distance_kms[i], self.distance_degrees[i], self.azimuths[i], self.antiAzimuths[
                i], self.parrivals[i], self.sarrivals[i], self.appro_line_distances[i], self.depths[i] = self._calarrivals(self.st[i])

        self.shift_values = np.zeros(len(self.r), dtype=np.float)
        if(self.y_axis_type == "Epicenter Distance"):
            self.shift_values = self.distance_degrees
        elif(self.y_axis_type == "Euclidean Distance"):
            self.shift_values = self.appro_line_distances
        elif(self.y_axis_type == "Depth"):
            self.shift_values = self.depths
        elif(self.y_axis_type == "Azimuth"):
            self.shift_values = self.azimuths

        if(np.max(self.shift_values) < self.y_range[1]):
            self.scale = (np.max(self.shift_values) -
                          np.min(self.shift_values))/15*self.scale
        else:
            self.scale = (self.y_range[1]-self.y_range[0])/15*self.scale

    def _plot_p(self):
        self.ax = self.canvas.fig.add_subplot(111)
        aligned_times = self.time+self.parrivals
        for i, value in enumerate(self.st):
            if(self.y_range[0] <= self.shift_values[i] <= self.y_range[1]):
                snr = self._calsnr(value, self.time[i], self.time[i]+self.parrivals[i], self.time[i]+(
                    self.parrivals[i]+self.sarrivals[i])/2)
                plotwaveform_obspy = value.slice(
                    aligned_times[i]-self.preset, aligned_times[i]+self.afterset)
                t = np.linspace(-self.preset, self.afterset,
                                plotwaveform_obspy.stats.npts)
                plotwaveform_data = plotwaveform_obspy.data
                self.ax.plot(t, plotwaveform_data*self.scale +
                             self.shift_values[i], color=str(self._snr_color(snr)))
        sx = self.sarrivals-self.parrivals
        sy = self.shift_values
        sx = [x for (y, x) in sorted(zip(sy, sx))]
        sy = [y for y in sorted(sy)]
        self.ax.plot(sx, sy, linestyle="--", color="b")
        self.ax.axvline(x=0, color="r", linestyle="--")

    def _plot_s(self):
        self.ax = self.canvas.fig.add_subplot(111)
        aligned_times = self.time+self.sarrivals
        for i, value in enumerate(self.st):
            if(self.y_range[0] <= self.shift_values[i] <= self.y_range[1]):
                snr = self._calsnr(value, self.time[i]+self.parrivals[i]+(
                    self.sarrivals[i]-self.parrivals[i])/2, self.time[i]+self.sarrivals[i], self.time[i]+self.sarrivals[i]+10)
                plotwaveform_obspy = value.slice(
                    aligned_times[i]-self.preset, aligned_times[i]+self.afterset)
                t = np.linspace(-self.preset, self.afterset,
                                plotwaveform_obspy.stats.npts)
                plotwaveform_data = plotwaveform_obspy.data
                self.ax.plot(t, plotwaveform_data*self.scale +
                             self.shift_values[i], color=str(self._snr_color(snr)))
        self.ax.axvline(x=0, color="b", linestyle="--")
