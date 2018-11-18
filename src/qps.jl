function QuadraticModel(qps::QPSData)
    QuadraticModel(qps.c, qps.Q, opHermitian(qps.Q), qps.A,
                   qps.lcon, qps.ucon, qps.lvar, qps.uvar, c0=qps.c0)
end
