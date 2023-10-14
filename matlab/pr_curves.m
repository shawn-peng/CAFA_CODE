
function curves = pr_curves(pred, ref, tau_range)
    n = size(pred, 1);
    m = size(tau_range, 1);
    precisions = zeros(n, m);
    recalls = zeros(n, m);
    for ti = 1:m
        tau = tau_range(ti);
        pos_pred = pred >= tau;
        tp_mat = pos_pred & ref;
        ntp = sum(tp_mat,2);
        npos_pred = sum(pos_pred,2);
        npos = sum(ref,2);
        precisions(:, ti) = ntp ./ npos_pred;
        %precisions(npos_pred==0, ti) = 0;
        recalls(:, ti) = ntp ./ npos;
    end
    curves = cell(n, 1);
    for i = 1:n
        curve = zeros(m, 2);
        curve(:, 1) = precisions(i, :);
        curve(:, 2) = recalls(i, :);
        curves{i} = curve;
    end
end