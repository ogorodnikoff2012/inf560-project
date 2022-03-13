COMPILE
---

    $ make

RUN
---

To run this project on the Polytechnique cluster, you need to
set up the environment:

    $ source set_env.sh

First method is to run the image processor directly:

    $ ./sobelf path/to/original/image.gif path/to/result.gif

We recommend using batch processing script. In order to use it,
you need to generate `filelist.txt` file with list of input images.
For example, it can be done with the folowing command:

    $ find images/original -name '*.gif' > filelist.txt

Next, you can run it with the following command:

    $ OMP=<number_of_threads> MPI=<number_of_nodes> ./run_test_from_list.txt

In order to perform benchmark, you can use the following script:

    $ ./run_with_params.txt


