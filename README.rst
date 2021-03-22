EO times series.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A brief demo of extracting time series for a point shapefile and writing as attributes. This is not exhaustive, simply what has been required of a recent project

It should be noted that the input rasters have already been averaged by NASA/USGS prior to the averaging below.

I have just provided cut down modules locally in the directory src. Provided you are using the notebook herein you can import the functions from the local files. 
Installation and Use
~~~~~~~~~~~~~~~~~~~~

Installing the required libs (there are not many) uses the conda system so ensure you have this first. Clone this repo and cd into the directory then...

.. code-block:: bash

conda env create -f eot_demo.yml

conda activate ndvi_demo

jupyter notebook

Then open the ipynb and cycle through the cells.

