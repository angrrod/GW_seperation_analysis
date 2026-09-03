# this short script builds the PSD for both the ET from the ET_D PSD and the cosmic_explorer_strain PSD.
import numpy as np

def Main():
    def _createPSDFile(ASD_file_name):
        #load realistic background and PSD
        asd_f, asd = np.loadtxt(ASD_file_name+".txt", unpack=True)  
        psd        = asd**2
        np.savetxt(ASD_file_name+"_PSD"+".txt", np.column_stack([asd_f, psd]))
        
    _createPSDFile("cosmic_explorer_strain")
    _createPSDFile("ET_D")
    
if __name__ == "__main__":
    Main()