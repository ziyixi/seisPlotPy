if(__name__ == "__main__"):
    from PyQt5 import QtWidgets
    from seisPlotPy.Main import MyApp
    import sys

    # Create GUI application
    app = QtWidgets.QApplication(sys.argv)
    form = MyApp()
    form.show()
    app.exec_()