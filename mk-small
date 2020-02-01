#! /bin/sh

DIR=$HOME/data
NEW=$HOME/data/small

# create folder if needed
[ -d "$NEW" ] || mkdir $NEW

# create TSV file
ls $DIR | fgrep .tsv | while read FILE
do
  head -500 < $DIR/$FILE > $NEW/$FILE
done

# copy files which are mentioned
cat $NEW/*.tsv | fgrep -- - | cut -c1-36 | while read FILE
do
  JSON=`cd $DIR && echo */*/$FILE.json`
  JSON_DIR=`dirname $JSON`
  [ -d "$NEW/$JSON_DIR" ] || mkdir -p "$NEW/$JSON_DIR"
  cp $DIR/$JSON $NEW/$JSON
done
