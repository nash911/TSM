# TSM
A modified version of the Traveling SalesMan problem


## Installation

Needs Python 3.6.7 or greater and pip3

```bash
$ sudo apt-get -y install python3 python3-pip
```

Then install the required packages

```bash
$ pip3 install -r requirements.txt
```


## Usage

From the project's root directory:

```bash
$ python3 tsm.py -i path/to/input
```

You can use the [-s|--start_city] argument to specify the index [1 to N] of the desired city to start the journey from, as follows

```bash
$ python3 tsm.py -i path/to/input -s <city_index>
```

To find the best route optimizing for the shortest duration of travel, instead of the lowest cost, you can use the [-t|--time] flag, as follows:

```bash
$ python3 tsm.py -i path/to/input -t
```

Project Organization
--------------------

    ├── README.md          <- The README file for end users.
    │
    ├── tsm.py             <- The main project file containing all the user interface, input validation, and most of the boilerplate code.
    │
    ├── dp.py              <- Contains the DP (Dynamic Programming) class, and the core DP solution for the TSM problem.
    │
    ├── utils.py           <- Subroutines for parsing input file(s) and plotting network graph for visualization are in this file.
    │
    ├── data_generator.py  <- Contains code for generating different kinds of random graphs for testing. This file is not part of the submitted solution,
    │                         but included for completeness as it is part of the git repo.
    │
     
