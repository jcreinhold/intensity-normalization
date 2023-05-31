=======
History
=======

2.2.4 (2023-05-31)
------------------

* Update to allow Python >=3.9

2.2.3 (2022-03-15)
------------------

* Revert error on different image shapes from ``RavelNormalize``; it is required!

2.2.2 (2022-03-15)
------------------

* Remove plural from ``Modality`` and ``TissueType`` enumerations.
* Update tutorials to use the ``Modality`` and ``TissueType`` enumerations.
* Remove error on different image shapes from ``RavelNormalize`` when registration enabled.

2.2.1 (2022-03-14)
------------------

* Update documentation to support modifications to Python API
* Update dependencies
* Remove incorrect warning from WhiteStripe normalization

2.2.0 (2022-02-25)
------------------

* Change backend to ``pymedio`` to support more medical image formats

2.1.4 (2022-01-17)
------------------

* Fix testing bugs in 2.1.3 and cleanup some interfaces

2.1.3 (2022-01-17)
------------------

* Cleanup Makefile and dependencies
* Add py.typed file

2.1.2 (2022-01-03)
------------------

* Updates for security

2.1.1 (2021-10-20)
------------------

* Fix warning about negative values when not passing in a mask
* Remove redundant word from Nyul normalize keyword arguments

2.1.0 (2021-10-13)
------------------

* Restructure CLIs for faster startup and improve handling of missing antspy
* add option to CLIs to print version

2.0.3 (2021-10-11)
------------------

* Improve Nyul docs and argument/keyword names

2.0.2 (2021-09-27)
------------------

* Fix and improve documentation
* Add an escape-hatch "other" modality
* Add peak types as modalities in KDE & WS

2.0.1 (2021-08-31)
------------------

* Save and load fit artifacts for LSQ and Nyul for both the CLIs and Python API

2.0.0 (2021-08-22)
------------------

* Major refactor to reduce redundancy, make code more usable outside of the CLIs, and generally improve code quality.

1.4.0 (2021-03-16)
------------------

* First release on PyPI.
