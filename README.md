README.md

## Fine Tune Model
0. Download 24h-48h of EMU data
1. Run Preprocess.py
     Montage -> filter ->
2. Fine tune pangolin on Spat115



## Real time Display
0. Data stream from ripple, multiple seconds to avoid filtering artifact
1. Montage -> filter (exactly same as preprocess.py)- > employ previous hist_equalization
 - load from metadata/scalingmetadaa/linear_interpolations_by_channel.pkl
 - each bin will have a value, so if data is coming in, all amplitudes will get mapped to a histogram bin 

For hist equalization - can check that the hist normalization has some wiggle
2. Run through pre-trained model in "eval" mode. 
3. Run through previously generated Pacmap
