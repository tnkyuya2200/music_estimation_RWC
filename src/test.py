import myfunctions as fn
import soundfile as sf

db = fn.Database("database/music_light.db")

m = db.load_Music_by_ID(0)
sf.write("hoge_+3.wav", m.y.T, m.sr)