
function model_results = get_model_prediction_results(model, targetcase, config)
    model_results = {};
    
    [~, model_results.model_ind] = ismember(model, config.model)
    model_results.fmax = targetcase.fmaxs(model_results.model_ind)
    model_results.model = config.model{model_results.model_ind}
    
    model_results.pred = load([config.pred_dir,model_results.model,'.mat']);
    model_results.pred = model_results.pred.pred;
    
    [~, ind] = ismember(targetcase.target, model_results.pred.object)
    model_results.pred = model_results.pred.score(ind, :);
    
    model_results.tau = all_taus(targetcase.bm_ind, model_results.model_ind)
    model_results.terms = oa.ontology.term(model_results.pred >= model_results.tau)
end
