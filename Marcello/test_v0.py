import pickle
import zlib

 
path = "/Users/marcello/Documents/Ael/bscw/Data_from_Michael/Marcello/"
filename = path + "______bscw_rom_fine_M08_alpha5_q150_presSolve_SST_test.pkl"

# with open("______bscw_rom_fine_M08_alpha5_q150_presSolve_SST_test.pkl", "rb") as handle:
with open(filename, "rb") as handle:
    comp_data = handle.read()
    data = zlib.decompress(comp_data)
    responses = pickle.loads(data)



