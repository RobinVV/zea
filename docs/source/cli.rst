Command line interface
================================

-------------------------------
Main CLI Command
-------------------------------

.. argparse::
   :module: zea.__main__
   :func: get_parser
   :prog: zea

-------------------------------
Data Copying Command
-------------------------------

.. argparse::
   :module: zea.data.__main__
   :func: get_parser
   :prog: python -m zea.data
   :nodefaultconst: