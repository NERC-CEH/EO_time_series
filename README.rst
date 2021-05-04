EO times series.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A brief demo of extracting time series for a point shapefile and writing as attributes. This is not exhaustive, simply what has been required of a recent project

I have just provided cut down modules locally in the directory src. Provided you are using the notebook herein you can import the functions from the local files.

This will likely be added to over time.  

Installation and Use
~~~~~~~~~~~~~~~~~~~~

Installing the required libs (there are not many) uses the conda system so ensure you have this first. Clone this repo and cd into the directory then...

.. code-block:: bash

conda env create -f eot_demo.yml

conda activate eot

jupyter notebook

Then open the ipynb and cycle through the cells.

Notes
~~~~~

You may note the functions spread the tasks client-side rather than server-side. Having written both, it becomes a case of moving the bottleneck between either client threads or file download from google if using server side parallelism. At present the file download is unreliable from google, and takes about the same length of time as spreading the process client side, hence the reliable clientside approach is used. Improvements on this welcome!
