module QuadraticModelsQPSReaderExt

import QuadraticModels
import QPSReader

function QuadraticModels.QuadraticModel(qps::QPSReader.QPSData, x0 = zeros(qps.nvar))
  QuadraticModels.QuadraticModel(
    qps.c,
    qps.qrows,
    qps.qcols,
    qps.qvals,
    Arows = qps.arows,
    Acols = qps.acols,
    Avals = qps.avals,
    lcon = qps.lcon,
    ucon = qps.ucon,
    lvar = qps.lvar,
    uvar = qps.uvar,
    c0 = qps.c0,
    x0 = x0,
  )
end

function QuadraticModels.RegularizedQuadraticModel(qps::QPSReader.QPSData, x0 = zeros(qps.nvar); σ = 0.0)
  model = QuadraticModels.QuadraticModel(qps, x0)
  QuadraticModels.RegularizedQuadraticModel(model, σ)
end

end
