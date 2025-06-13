[![CI](https://github.com/stan-dot/blue-histogramming/actions/workflows/ci.yml/badge.svg)](https://github.com/stan-dot/blue-histogramming/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/stan-dot/blue-histogramming/branch/main/graph/badge.svg)](https://codecov.io/gh/stan-dot/blue-histogramming)
[![PyPI](https://img.shields.io/pypi/v/blue-histogramming.svg)](https://pypi.org/project/blue-histogramming)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# blue_histogramming

Tying together bluesky plans with STOMP messages and Davidia HDF streaming

`mkdir /tmp/rabbitmqdata` for the devcontainer to work ok.

This is where you should write a short paragraph that describes what your module does,
how it does it, and why people should use it.

Source          | <https://github.com/stan-dot/blue-histogramming>
:---:           | :---:
PyPI            | `pip install blue-histogramming`
Docker          | `docker run ghcr.io/stan-dot/blue-histogramming:latest`
Releases        | <https://github.com/stan-dot/blue-histogramming/releases>

This is where you should put some images or code snippets that illustrate
some relevant examples. If it is a library then you might put some
introductory code here:

```python
from blue_histogramming import __version__

print(f"Hello blue_histogramming {__version__}")
```

Or if it is a commandline tool then you might put some example commands here:

```
python -m blue_histogramming --version
```


## Useful links

<https://github.com/LukasMasuch/streamlit-pydantic>

<https://discuss.streamlit.io/t/fastapi-backend-streamlit-frontend/55460>

<https://docs.streamlit.io/develop/tutorials/databases/postgresql>

<https://docs.streamlit.io/deploy/tutorials/kubernetes>

<https://www.rabbitmq.com/tutorials/tutorial-one-python>

<https://docs.bytewax.io/stable/guide/getting-started/join-example.html>

<https://github.com/bytewax/bytewax>


## from the spec meeting

todo add mulitple clients too
<https://fastapi.tiangolo.com/advanced/websockets/#handling-disconnections-and-multiple-clients>

snaking paths, pattern growing on the screen
with a pre-defined sum over rectangular region

so let's pick 3 region

to demonstrate 4 of those pltos at the same time

4th one is the total

bottlenecked on the stage speed - in step scanning
in fly scanning

from streamdatum, from stomp serialization

from run engine serialization of the scan spec

JSON - serialized version of the scan spec
to turn back

summary at the bottom?

button to run again?

also show the raw image

then a faster camera - stages already can fly scanning

probably 5 frames every half a second = 10Hz

the snake frames are 100 by 100 maps likely

10Hz will be the fastest we can see enough light

but thne 10 minute scan?

2 minutes as expected duration

after clicking a button

we see a pixel with intensity

we histogram them in terms of maximum intensity

the initial pixels need to get dimmer

we update live the reference value

davidia - davidia shoudl already do the histogramming

the sum - numpy array?

## pvs

dcam1 or dcam2

<https://gitlab.diamond.ac.uk/controls/containers/beamline/bl01c/-/blob/main/services/bl01c-di-dcam-01/config/ioc.yaml?ref_type=heads>

making aravis camera
and caget
<https://gitlab.diamond.ac.uk/controls/containers/beamline/bl01c/-/blob/main/services/bl01c-di-dcam-02/config/ioc.yaml?ref_type=heads>

pattern detector as argument to the count plan
that will generate stream resource for the data, for the sample
and it will provide something

reading library that uses h5py

we need to use a pattern generator detector

call a reading library with stream resource

### detector data

> > > f['entry']['instrument']['detector']['data'].shape
> > > (4, 1216, 1936)
> > > f['entry']['instrument']['detector']['data'][:2]
> > > array([[[1, 1, 1, ..., 1, 1, 1],

        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        ...,
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1]],

       [[1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        ...,
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1]]], shape=(2, 1216, 1936), dtype=uint8)

> > >


## after 7 iterations

started with simple data streaming numpy arrays .

https://docs.h5py.org/en/latest/swmr.html

```python
import h5py

f = h5py.File("swmr.h5", "r", libver="latest", swmr=True)
dset = f["data"]
while True:
    dset.id.refresh()
    shape = dset.shape
    print(shape)
```


also used an outdated 'pyinotify' library to watch for changes in the target file.
Fortunately manta is an AravisDetector which supports swmr mode.
The image 

Then I tried the watchdog library, but looks like listening to bluesky events is a better solution for data update being at the right moment.

Then iteration three was running segments from a dataset


This would likely be a standalone server, kind of like h5grove but more involved. This on its own requires a separate package with python copier template.

The features desired there would be a synthesis of various ideas prototyped so far:
- davidia ws for the data so that it can be displayed
- fake data streaming as part of a scripts directory for testing, inside the src one? or to not export it to be honest
- simple listener too just to listen to events 
- using stateless setup with redis - like number 6
- provide random data too at selected endpoints
- provide simple listener - into websockets to stream just all events, and create a log. 
- provide plan-specific listener that works in collaboration with 'launch demo' endpoint

The only rejected feature would be the use of a watchdog / pyinotify UNLESS we want to preview all the hdf that happened in a directory, not just the new events. 

the UI could like a websockets - to - file tree and then next the hdf tree for a selected file then finally on the right the panel with davidia visualization

Let's for now do per-login data sessions with unique server-assigned IDs - anonymous sessions - however access only restricted to a directory defined in an environment variable.

Need to make tags at api routes to group them.


## how to get mock data
it's logged automatically by blueapi, just run the plan and get the logs from there.
# normalize based on events
# xvals = [e["data"][motor_names[-1]] for e in events]
# yvals = [e["data"][main_detector_name] for e in events]
# # normalize
# xvals = [x - xvals[0] for x in xvals]

# # todo consider a dataframe instead? maybe with polars https://pola.rs/
results = []
# # https://docs.pydantic.dev/latest/examples/files/#csv-files

# https://docs.python.org/3/library/errno.html

"""
Callback listener that processes collected documents and
fits detector data with curve :
<li>Single curve for 1-dimensional line scan,
<li> N curves for grid scans with shape NxM (M points per curve).

Uses scipy curve_fit function for curve fitting
fit_function -> function to be used during fitting
fit_bounds -> range for each parameter to be used when fitting.
    A tuple of (min, max) value for each parameter.
    e.g. for parameters a, b,c : ( (min a, max a), (min b, max b), (min c, max c))
"""
