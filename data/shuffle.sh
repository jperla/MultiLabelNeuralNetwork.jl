#!/bin/sh

cat data/emotions/emotions-train.csv | perl -MList::Util -e 'print List::Util::shuffle <>' > ./data/emotions/emotions-train.csv.shuffled
cat data/scene/scene-train.csv | perl -MList::Util -e 'print List::Util::shuffle <>' > ./data/scene/scene-train.csv.shuffled
cat data/yeast/yeast-train.csv | perl -MList::Util -e 'print List::Util::shuffle <>' > ./data/yeast/yeast-train.csv.shuffled
cat data/reuters/reuters-train.csv | perl -MList::Util -e 'print List::Util::shuffle <>' > ./data/reuters/reuters-train.csv.shuffled
