entity_sample=test_qIDs.tsv
input_file=latest-truthy_01012022_cleaned.txt
output_file=gender_facts_test_10.txt
echo "Started extracting at:"
date
for qid in $(cat $entity_sample)
do
	echo $qid" - start"
	grep -E "^$qid P21 Q" $input_file >> $output_file
	echo " - done"
done
echo "Finished extraction at:"
date
