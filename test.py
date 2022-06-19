import myfunctions as fn
import soundfile

db = fn.Database("src/music_light.db")
m = db.load_Music_by_ID(209)
soundfile.write("vocals.wav", m.esti_vocals.T, m.sr, format="WAV")
soundfile.write("acc.wav", m.esti_acc.T, m.sr, format="WAV")
