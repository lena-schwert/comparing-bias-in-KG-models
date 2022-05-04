echo "Started extracting at:"
date

zgrep "<http://www.wikidata.org/prop/direct/P" latest-truthy_01012022.nt.gz | sed 's/<http:\/\/www.wikidata.org\/entity\///g' | sed 's/<http:\/\/www.wikidata.org\/prop\/direct\///g' | sed 's/.$//'| sed 's/>//g' > latest-truthy_01012022_cleaned.txt
wc -l latest-truthy_01012022_cleaned.txt

echo "Finished extraction at:"
date
