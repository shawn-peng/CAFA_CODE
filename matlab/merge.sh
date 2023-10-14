
cat hpo_HUMAN_type1.txt > hpo_all_type1.txt

cat bpo_all_type1.txt cco_all_type1.txt mfo_all_type1.txt hpo_all_type1.txt |sort |uniq > xxo_all_type1.txt
cat bpo_all_type2.txt cco_all_type2.txt mfo_all_type2.txt |sort |uniq > xxo_all_type2.txt

cat bpo_all_type1.txt bpo_all_type2.txt |sort |uniq > bpo_all_typex.txt
cat cco_all_type1.txt cco_all_type2.txt |sort |uniq > cco_all_typex.txt
cat mfo_all_type1.txt mfo_all_type2.txt |sort |uniq > mfo_all_typex.txt
cat hpo_all_type1.txt > hpo_all_typex.txt

cat xxo_all_type1.txt xxo_all_type2.txt |sort |uniq > xxo_all_typex.txt


