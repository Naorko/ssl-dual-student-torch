#!/usr/bin/env bash

# Prepare semisupervised datasets needed in the experiments

SCRIPT=labels/bin/create_balanced_semisupervised_labels.sh

create ()
{
    DATADIR=images/${DATANAME}
    for LABELS_PER_CLASS in ${LABEL_VARIATIONS[@]}
    do
        LABELS_IN_TOTAL=$(( $LABELS_PER_CLASS * $NUM_CLASSES ))
        echo "Creating sets for $DATANAME with $LABELS_IN_TOTAL labels."

        LABEL_DIR=labels/${DATANAME}
        mkdir -p $LABEL_DIR
        $SCRIPT $DATADIR $LABELS_PER_CLASS > $LABEL_DIR/${LABELS_IN_TOTAL}.txt
    done
}


# shellcheck disable=SC2054
DATANAME=cifar100
NUM_CLASSES=100
LABEL_VARIATIONS=(50 100 200 300 400)
create

DATANAME=cinic10
NUM_CLASSES=10
LABEL_VARIATIONS=(5000 10000 20000 30000 40000)
create

DATANAME=cub200
NUM_CLASSES=200
LABEL_VARIATIONS=(5 10 20 30 40)
create

DATANAME=mnist
NUM_CLASSES=10
LABEL_VARIATIONS=(500 1000 2000 3000 4000)
create
