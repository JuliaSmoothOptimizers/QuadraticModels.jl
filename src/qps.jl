using QPSReader
function QuadraticModel(qps::QPSData, x0...)
    if x0 == ()
        QuadraticModel(qps.c, qps.qrows, qps.qcols, qps.qvals, Arows=qps.arows,
                       Acols=qps.acols, Avals=qps.avals, lcon=qps.lcon,
                       ucon=qps.ucon, lvar=qps.lvar, uvar=qps.uvar, c0=qps.c0)
    else
        QuadraticModel(qps.c, qps.qrows, qps.qcols, qps.qvals, Arows=qps.arows,
                       Acols=qps.acols, Avals=qps.avals, lcon=qps.lcon,
                       ucon=qps.ucon, lvar=qps.lvar, uvar=qps.uvar, c0=qps.c0,
                       x0=x0[1])
    end
end
