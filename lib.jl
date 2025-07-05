function _train_model(model, fold)
    trn, val = fold
    train!(model; training = trn)
    pred = predict(model, features(model)[:, val]; threshold = false)
    return pred, val
end 

function _compute_curve(pred, val, model, xmet, ymet, thres)
    xs, ys = Float32[], Float32[]
    for τ in thres
        Cv = ConfusionMatrix(pred, labels(model)[val], τ)
        push!(xs, xmet(Cv))
        push!(ys, ymet(Cv))
    end 
    return xs, ys 
end

function roc(model, fold::ConfusionMatrix, thres = LinRange(0,1,100))
    pred, val = _train_model(model, fold)
    _compute_curve(pred, val, model, fpr, tpr, thres)
end 

function roc(model, folds::Vector{<:ConfusionMatrix}, thres = LinRange(0,1,100))
    [roc(model, fold, thres) for fold in folds]
end 


function pr(model, fold::ConfusionMatrix, thres = LinRange(0,1,100))
    pred, val = _train_model(model, fold)
    _compute_curve(pred, val, model, ppv, tpr, thres)
end
function pr(model, folds::Vector{<:ConfusionMatrix}, thres = LinRange(0,1,100))
    [pr(model, fold, thres) for fold in folds]
end

rocauc(args...) = SDT.SDeMo.auc(roc(args...)...)
prauc(args...) = SDT.SDeMo.auc(pr(args...)...)