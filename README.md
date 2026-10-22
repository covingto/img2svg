# img2svg

## Problem

Are you frustrated that AI can get you 90% to the diagram you want in 10 seconds and then make you spend 2 hours getting to 99%? Use this Docker web app locally (or host it if you want) to convert those diagrams to svg and edit it yourself in minutes.

## Install and Setup

To build the container you will need docker installed. From there simply do:

```
docker build -t img2svg .
```

Then you can run: 

```
docker run --rm -p 8090:8000 img2svg
```

Open your favorite browser at http://localhost:8090 and you will see the web app
where you can drop in an image and the container will convert it to an SVG.

## Improving Object Segmentation

### AI prompting

We assume that you are generating images from AI. Prompting the AI to generate
clearly distinct objects outlined with a black stroke (like a cartoon),
encouraging it to use solid fills in lighter hues helps the threshold
algorithms to better separate objects.

Other tips are to have the AI omit text, numbers, or other clutter that might
confuse the segmentation algorithm. All shapes will be converted as paths,
so any text that is converted will not be svg text and not editable using your
editor's text tools. You may also encourage the AI to make asset libraries
for you which you can then convert and assemble into your own diagrams. If there
are small objects in the image that do not render well, you can ask the AI to
create those as separate assets, do the conversion, and then incorporate those
into your svg image separately.

### Threshold tuning

The default Otsu threshold works most of the time to generate a clean svg
output. You can use the middle segmentation view to judge how well changes
to the thresholds work. If you turn off Otsu thresholding the system will
instead use the saturation and value settings to attempt to segment the image
by color, using those settings to generate a binary mask that attempts to find
dark outlines. 

The segmentation view will automatically update as you alter the settings.

Once you have a segmentation that you like you can press the 'Process Image'
button and a new svg will be generated.

## Download

Simply press the 'Download SVG' button and an svg file will be downloaded to
your computer and you can open it in your favorite svg editing software (I like
Inkscape). 


