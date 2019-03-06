.. _Config:

*************
Configuration
*************


The easiest way to configure your tests is by defining a ``.mlrc`` file. In it you can set up all the options
required to successfully run your tests.


.. code-block:: console

   # .mlrc to control testml
   [run]
   metrics = accuracy_score, recall_score
   data = extras/test.csv
   loader = joblib
   model = mdl.joblib
   [report]
   failure = 0
   report = console


The ``[run]`` section mostly deals with configuration related to how the model will be loaded and evaluated.
The ``[report]`` section instead is concerned with the ability to set thresholds for acceptance, and how
a report should be generated.


.. seealso:: :ref:`CliAnchor` for how to set these options through the cli.

