input_file=latest-truthy_01012022_cleaned.txt
output_file=all_gender_facts_test_23032022_2.txt
echo "Started extracting at:"
date

grep "P21 Q" $input_file >> $output_file

echo "Finished extraction at:"
date
