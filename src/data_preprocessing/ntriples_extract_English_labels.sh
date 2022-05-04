echo "Started extracting at:"
date

zgrep "<http://schema.org/name>" latest-truthy_01012022.nt.gz | grep "@en" | sed 's/<http:\/\/www.wikidata.org\/entity\///g' | sed 's/<http:\/\/schema.org\/name> "//g' | sed 's/.$//'| sed 's/>//g' | sed 's/"//g'

echo "Finished extraction at:"
date
