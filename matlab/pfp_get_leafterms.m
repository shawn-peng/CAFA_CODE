function leaves = pfp_get_leafterms(oa)
leafann = pfp_leafannot(oa);

n = size(oa.annotation, 1);

leaves = cell(n);

for i = 1:n
    x = oa.ontology.term(leafann(i,:));
    if size(x,1) == 1
        if x.id == '
    leaves{i} = x;
end

end
