## To use

- Follow [sleap-anipose installation guide](https://github.com/talmolab/sleap-anipose/blob/main/README.md) 

- Clone the repo.

- Simply set the params in the notebook file provided and then run it.

## Note

The code currently assumes that the input videos are videos with multiple views fused together (obtained with OBS). (See the video example.) Separate views can be supported if you manually arrange them in the folder structure as follows:

    
    ROOTPATH
        /SA_calib/views1/calibration_images/SA_calib-view1-calibration.mp4
        /SA_calib/views1/calibration_images/SA_calib-view2-calibration.mp4
        ...
        
        /SD-{CUSTOMNAME}/Videos/cam1/0.mp4
        /SD-{CUSTOMNAME}/Videos/cam2/0.mp4
        ...
        

## Links

[Sleap-anipose](https://github.com/talmolab/sleap-anipose)    [DANNCE](https://github.com/spoonsso/dannce/)
[SDANNCE](https://github.com/tqxli/sdannce)            [Label3D](https://github.com/diegoaldarondo/Label3D)
