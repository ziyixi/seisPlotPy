import numpy as np
import obspy
import glob
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

import seisPlotPy.interface as interface
from seisPlotPy.obspyPssac import Pssac


class MyApp(QMainWindow, interface.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.config = {}
        self.catalogName = ""
        self.eventid = np.array([])
        self.UTCDateTime = np.array([])
        self.stationid = np.array([])
        
        self.baseurl = "./data/"
        self.stationName="./station.dat"
        self.catalogName="./catalog.dat"

        self.setupUi(self)
        self.bindSignal()

    def bindSignal(self):
        self.FreqMaxSlider.valueChanged.connect(self.vuelike)
        self.FreqMinSlider.valueChanged.connect(self.vuelike)
        self.modeComboBox.activated[str].connect(self.vuelike)

        self.loadCatalogButton.clicked.connect(self.loadCatalog)
        self.loadStationButton.clicked.connect(self.loadStation)
        self.directoryButton.clicked.connect(self.loadDirectory)
        self.plotButton.clicked.connect(self.plotPssac)

    def vuelike(self):
        sender = self.sender()
        if(sender == self.FreqMaxSlider):
            self.freqmaxLabel.setText(
                f"max frequency {self.FreqMaxSlider.value()}")
        elif(sender == self.FreqMinSlider):
            self.freqminLabel.setText(
                f"min frequency {self.FreqMinSlider.value()}")
        elif(sender == self.modeComboBox):
            text = self.modeComboBox.currentText()
            if(text == "Event"):
                self.stationComboBox.clear()
                self.stationComboBox.update()
                self.eventComboBox.clear()
                self.eventComboBox.addItems(list(self.catalog.id.values))
                self.eventComboBox.update()

            elif(text == "Station"):
                self.eventComboBox.clear()
                self.eventComboBox.update()
                self.stationComboBox.clear()
                self.stationComboBox.addItems(list(self.stationid))
                self.stationComboBox.update()

    def loadCatalog(self):
        self.catalogName = QFileDialog.getOpenFileName(
                self, 'Load Catalog', './')[0]
        print("catname",self.catalogName)
        if(not self.catalogName):
            return

        self.catalog = pd.read_table(self.catalogName, names=[
                                     "id", "date", "time"], usecols=(0, 1, 2), sep="\s+")
        self.catalog["UTCDateTime"] = np.nan
        for i in range(self.catalog.shape[0]):
            self.catalog.loc[i, "UTCDateTime"] = obspy.UTCDateTime(
                f"{self.catalog.loc[i,'date']} {self.catalog.loc[i,'time']}")

        self.eventid = self.catalog.id.values
        self.UTCDateTime = self.catalog.UTCDateTime.values

        if(self.modeComboBox.currentText() == "Event"):
            self.eventComboBox.clear()
            self.eventComboBox.addItems(list(self.eventid))
            self.eventComboBox.update()

    def loadStation(self):
        self.stationName = QFileDialog.getOpenFileName(
                self, 'Load Station', './')[0]
        print("stname",self.stationName)
        if(not self.stationName):
            return

        self.station = pd.read_table(self.stationName, names=[
                                     "id"], usecols=(0,), sep="\s+")
        self.stationid = self.station.id.values

        if(self.modeComboBox.currentText() == "Station"):
            self.stationComboBox.clear()
            self.stationComboBox.addItems(list(self.station.id.values))
            self.stationComboBox.update()

    def loadDirectory(self):
        self.baseurl = QFileDialog.getExistingDirectory(
                self, 'Load Waveform Directory', './')
        print("diname",self.baseurl)
        if(not self.stationName):
            return

    def updateStations(self):
        eventSelected=self.eventComboBox.currentText()
        stations=glob.glob("data/"+eventSelected+"/*R")
        result=[i.split(".")[1] for i in stations]
        return result


    def plotPssac(self):
        if(self.modeComboBox.currentText()=="Event"):
            self.stationid=self.updateStations()

        self.config = {
            "preset": int(self.preSetLineEdit.text()),
            "afterset": int(self.afterSetLineEdit.text()),
            "model": self.oneDModelComboBox.currentText(),
            "scale": float(self.scaleLineEdit.text()),
            "texted": self.textTypeComboBox.currentText(),
            "y_range": [float(self.yMinLineEdit.text()), float(self.yMaxLineEdit.text())],
            "global_normal": self.globalNormCheckBox.isChecked(),
            "aligned_phase": self.alignedPhaseComboBox.currentText(),
            "y_axis_type": self.yaxisTypeComboBox.currentText(),
            "filter_band": [self.FreqMinSlider.value(), self.FreqMaxSlider.value()],
            "direction": self.directionComboBox.currentText(),
            "mode": self.modeComboBox.currentText(),
            "eventid": self.eventid,
            "UTCDateTime": self.UTCDateTime,
            "stationid": self.stationid,
            "station_selected": self.stationComboBox.currentText(),
            "event_selected": self.eventComboBox.currentText()
        }

        self.pssacObject = Pssac(self.PlotRegion.canvas, config=self.config)
        self.pssacObject.show()

    def test(self):
        # TODO: delete test
        pass
        # print(type(self.yMinLineEdit.text()))
        # print(self.FreqMinSlider.value())
        # print(self.FreqMaxSlider.value())
        # print(self.yMinLineEdit.text())
        # print(self.yMaxLineEdit.text())
        # print(self.preSetLineEdit.text())
        # print(self.afterSetLineEdit.text())
        # print(self.scaleLineEdit.text())
        # print(self.textTypeComboBox.currentText())
        # print(self.oneDModelComboBox.currentText())
        # print(self.modeComboBox.currentText())
        # print(self.alignedPhaseComboBox.currentText())
        # print(self.globalNormCheckBox.isChecked())
        # pssacObject=Pssac(self.PlotRegion.canvas)
        # pssacObject.test()


if(__name__ == "__main__"):
    from PyQt5 import QtWidgets
    import sys

    # Create GUI application
    app = QtWidgets.QApplication(sys.argv)
    form = MyApp()
    form.show()
    app.exec_()
