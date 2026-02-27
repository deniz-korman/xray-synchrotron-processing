# Synchrotron x-ray image processing (for jumping spider eyes)

For more information on the original codebase, with tools to facilitate ML-based tracking, and QA-tools for tracking motion and proportionality of tracks, visit https://github.com/jacksonmichaels/image-and-path-processing-toolkit

## Config Files

Config files specify how the pipeline functions, and what images to process.

Once a config file is established, you can process all specified images with a single call such as:

`python .\video_process.py -c .\configs\sample_salticid_tbf.yaml -r`

*It is important you include -r for recursive otherwise it will not search subfolders for images.*

The processing pipelines described in our paper can be achieved with the following config files. They capture 'static' and 'dynamic' features respectively
1. **sample_salticid_stills.yaml**
   - The x-ray images specified in the ```image_folder``` are imported, and the x-ray background noise footage specified in ```flat_folder``` is filtered out.
   - The footage in this format is returned in mp4 format, as well as a png still at every ```num_images``` (500) frames highlighting external structures.

2. **sample_salticid_tbf.yaml**
   - The x-ray images specified in the ```image_folder``` are imported, and the x-ray background noise footage specified in ```flat_folder``` is filtered out.
   - All static elements in footage are filtered by dividing each frame by an average of the next 10 frames (1 sec period), highlighting moving elements.
   - The moving elements (internal eye-structures responding to visual stimuli) are further enhanced by applying a temporal bilateral filter in both the spatial and temporal domain.
        - Effective in boosting contrast in contiguous structures with defined edges, which work well for the eye-tubes of jumping spiders.

#### Example config file

```
# File Paths
image_folder: [PATH TO IMAGES]
flat_folder: [PATH TO FLATS] # X-ray background noise images
output_name: "internal_morphology_dynamics"

# Video Settings
## what frame to begin processing on
vid_start_index: 0

## number of images to read in. Important note is that this does not account for stride. a num_images of 500 and stride of 5 will result in a 100 frame video
## limiting num_images rather than running the full image-set can be useful to prevent crashes
num_images: 500

## stride to use when looking at images.
video_stride: 5

## useful if your videos are RGB but need to be grayscale
convert_to_gray: False

# Export a .png file with the first frame of every video
indiv_thumbnails: False 

## this crops the input videos to reduce memory usage and make it easier to view
crop: [
  [500,-1],
  [800,-1]
]

# Pipeline Stages
## This is how the pipeline is created in both methods. This should be a list of stages each one an object with a name and any required parameters. For details on methods look below.
pipeline:
  [
    {
      # Name of method. This needs to match exactly the name of the method
      # mean_divide_video divides each frame by the average of the following <params> number of frames
      name: mean_divide_video,
      # Any positional arguments (for example, how many leading frames to average and divide by)
      params:
        [10],
      # keyword arguments. for example the fourier_masker_center method takes a number of named inputs
      kwargs: {}
    },
    {
      # temporal_bilateral_filter applies a temporal bilateral filter in the spatial and temporal domain
      # enhancing edges, boosting the visibility of the internal eye-structures in the original use case.
      name: temporal_bilateral_filter,
      kwargs: {},
      params: []
    }
  ]

```


Most parameters can be specified both in the config file, as well as through command line arguments.

If a value is defined in both the config file and command line arguments it defaults to command line arguements. This is useful if you have a config file whos results you are happy with but would like to try on a different image set without changing the file. In both files the pipeline is customized but before the pipeline is run all images / videos / tracking results are read into memory. Then after the pipeline is run a video will be created based on the output name, you do not need to explicitly add that as a stage.

---
## Video Process

##### Command line args:

- config (-c)
    - path to config file, more details on config files below
- input (-i)
    - path to folder containing input images
- output(-o)
    - path to store processed videos 
- flat (-f)
    - path to flat images (if they are used). As a note they are not required for most pipelines as the mean divide server the same purpose.
- no_flat (-nf)
    - flag to pass when not using a flat image
- begin (-b)
    - frame number to begin processing. Useful if you have a long series of images and want a subset or have garbage data at the beginning of the sequence.
- num_images (-n)
    - number of images to read in from disk
- stride (-s)
    - stride of images used in processing, often times a stride of 2 helps amplify motion and improve filter quality.
- crop (-cr)
    - pixel values for cropping input images. helpful if images are large but only a small region is important. Also very helpful for limiting memory usage to allow longer sequences.
- recursive (-r)
    - add this flag for the pipeline to be run on all subfolders of input (for instance if you have multiple videos to run at once).

### Image and Video Processing Functions

#### Video Functions

1. **process_video(vid, func, args, kwargs)**: 
   - Applies a specified function to each frame of the video.
   - Useful for batch processing of video frames with custom image processing functions.

2. **write_video(vid, name, folder="videos")**:
   - Writes a video file from an array of frames.
   - Useful for saving processed videos.

#### Image Functions

1. **cv_denoise(img, strength=2)**:
   - Applies OpenCV's fastNlMeansDenoising to an image.
   - Reduces noise in images.

2. **blur(img, size=3)**:
   - Applies Gaussian blur to an image.
   - Useful for smoothing images.

3. **sharpen(img, size=3)**:
   - Sharpens an image using a Gaussian filter.
   - Enhances details in images.

4. **median_filter(img, size=3)**:
   - Applies a median filter.
   - Effective for removing salt-and-pepper noise.

5. **detect_edge(img, image_mul=1)**:
   - Detects edges in an image using PIL's FIND_EDGES filter.
   - Useful for highlighting edges.
   - image_mul scaled the entire image, useful if the images values are too low for the filter to detect.

6. **threshold_img(img, threshold=[0.1,0.8])**:
   - Applies a binary threshold to an image.
   - Useful for isolating elements based on intensity.

7. **modify_contrast(img, factor=1.5, img_mul=100)**:
   - Adjusts the contrast of an image.
   - Enhances the visual appearance or highlights features.

8. **laplacian(vid)**:
    - Applies a Laplacian filter to a video.
    - Enhances edges in video frames.

10. **mean_divide_video(vid, n_frames)**:
    - Divides each frame of the video by the mean of a number of surrounding frames.
    - Useful for normalizing brightness variations.

11. **mean_divide_video_bidir(vid, n_frames_f, n_frames_b)**:
    - Applies bidirectional mean division on video frames.
    - Helps in correcting illumination differences.
    - Very similar to mean divide method but gathers the mean from before and after the given frame

12. **mix_videos(vid_a, vid_b, mix_coef)**:
    - Mixes two videos based on a mixing coefficient.
    - Useful for combining features or effects from two different videos.

13. **temporal_bilateral_filter(vid)**:
    - Applies a bilateral filter temporally across video frames.
    - Effective in noise reduction while preserving edges.

#### Utility Functions

1. **norm_vid(images)**:
   - Normalizes video frames.
   - Ensures uniform brightness and contrast across frames.

2. **frame_norm(vid)**:
   - Normalizes each frame of a video independently.
   - Useful for correcting variations within each frame.

3. **std_normalize(vid, stds=1)**:
   - Normalizes a video based on standard deviation.
   - Balances the brightness and contrast based on statistical measures.

4. **normalize(vid, max_val=1)**:
   - Normalizes a video to a specified maximum value.
   - Ensures a consistent scale across video frames.

5. **mul(vid, coef)**:
   - Multiplies each frame of the video by a coefficient.
   - Useful for adjusting the intensity or brightness