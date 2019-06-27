#!/usr/bin/env bash
mkdir data
cd data
wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
tar zxvf rt-polaritydata.tar.gz
rm rt-polaritydata.tar.gz
mv rt-polaritydata/* .
rm -rf rt-polaritydata