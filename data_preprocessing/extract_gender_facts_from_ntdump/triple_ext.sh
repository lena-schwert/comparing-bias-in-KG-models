entity_sample=The qid_file
input_file=The cleaned ntriples file
output_file=The output filtered triples file
for qid in $(cat $entity_sample)
do
	echo $qid" - start"
	grep -E "^$qid	P" $input_file >> $output_file
	echo " - done"
done
