#=

This entire file was written by iterating with Claude.

=#

function brent(g, x::AbstractArray, params...; xa, xb, tol=eltype(x)(1e-13), maxiter=500, check_every=1)
    g!(_y, x) = (_y .= g(x, params...); _y)
    y = g(x, params...)
    sol = brent!(g!, y, x, params...; xa, xb, tol, maxiter, check_every)
    return 
end

"""
    brent(g, xa, xb; tol, maxiter, check_every) -> (xmax, gmax, converged)

Find the minimum of vectorized `g` over `[xa[i], xb[i]]` for each element i.

`g` must accept xa vector input and return xa vector of the same backend.

`check_every`: iterations between convergence checks (default 1). Higher
values reduce host-sync overhead on GPU for expensive `g`.
Set to 0 to skip mid-loop checks (one final check always runs at return).

DOES NOT SUPPORT SCALAR EVALUATION!

Note no batchdim here because 1D so any direction is used
"""
@generated function brent!(
    g!::Function,
    y,
    x::AbstractArray{T},
    params...;
    xa,
    xb,
    tol         = T(1e-13),
    maxiter     = 500,
    check_every = 1,
) where {T}
    # This weirdness (generated) is needed for compatibility with Metal
    # which literally cannot compile even if we immediately cast compile 
    # time constant to Float32.
    CGOLD = T(0.3819660112501051)
    return quote
        @assert length(xa) == length(xb)

        xa  = copy(xa)
        xb  = copy(xb)

        @. x  = xa + $CGOLD * (xb - xa)
        gx = similar(y)
        g!(gx, x, params...)
        w  = copy(x);  gw = copy(gx)
        v  = copy(x);  gv = copy(gx)

        # Algorithm state
        d         = similar(xa); d .= zero(T)   # last step taken
        e         = similar(xa); e .= zero(T)   # step before last (NR: start at 0)
        converged = similar(xa, Bool); converged .= false

        # ── Scratch arrays: 10 float + 5 bool ────────────────────────────────
        #
        # Float scratch — reuse strategy noted inline:
        #   p     : holds r=(x-w)*(gx-gv) first, then p_final
        #   q     : holds q_final = 2*(q_raw - r)
        #   d_para: parabolic step p/q; reused as bound in bracket update
        #   gs_e  : golden-section sub-interval (xa-x or xb-x)
        #   d_chos: chosen step pre-clamp; also written to e
        #   mid   : (xa+xb)/2; reused for tol2 expression inline
        #   tol1  : tol*|x|+eps(T); negated inline where -tol1 needed
        #   u     : trial point x+d; reused as u_eval (masked in-place)
        #   y    : g!(y, u_eval); masked in-place for converged lanes
        #
        # Bool scratch:
        #   para_ok : parabola accepted flag; reused for use_edge mask
        #   x_right : x>=mid; reused as u_lt_x then update_a
        #   is_best : y<gx; reused as gu_best for bracket update
        #   is_2nd, is_3rd: point bookkeeping
        mid     = similar(xa)
        tol1    = similar(xa)
        p       = similar(xa)
        q       = similar(xa)
        d_para  = similar(xa)
        gs_e    = similar(xa)
        d_chos  = similar(xa)
        u       = similar(xa)
        para_ok = similar(xa, Bool)
        x_right = similar(xa, Bool)
        is_best = similar(xa, Bool)
        is_2nd  = similar(xa, Bool)
        is_3rd  = similar(xa, Bool)

        for iter in 1:maxiter

            @. mid  = (xa + xb) / 2
            @. tol1 = tol * abs(x) + $eps(T)

            if check_every > 0 && mod(iter, check_every) == 0
                @. converged = abs(x - mid) <= 2*tol1 - (xb - xa) / 2
                all(converged) && break
            end

            # ── Parabolic interpolation ───────────────────────────────────────
            # Compute r into p, q_raw into q, then:
            #   q_final = 2*(q_raw - r)        [update q while p=r]
            #   p_final = (x-v)*q_final/2 + r*(w-v)  [algebraic rearrangement,
            #             avoids storing r separately; verified by substitution]
            @. p     = (x - w) * (gx - gv)              # r → p
            @. q     = (x - v) * (gx - gw)              # q_raw → q
            @. q     = 2 * (q - p)                       # q_final (p=r still valid)
            @. p     = (x - v) * q / 2 + p * (w - v)   # p_final

            # Flip sign so q > 0 (NR convention), then abs(q)
            @. p     = ifelse(q > zero(T), -p, p)
            @. q     = abs(q)

            # e holds step from 2 iters ago (saved before overwrite)
            # para_ok temporarily holds the acceptance test result
            @. para_ok = (abs(e) > tol1) &
                         (abs(p) < abs(q) * abs(e / 2)) &
                         (p > q * (xa - x)) &
                         (p < q * (xb - x))

            # Parabolic step (guarded against q==0)
            @. d_para  = ifelse(q != zero(T), p / q, zero(T))

            # Golden-section step into the larger sub-interval
            @. x_right = x >= mid
            @. gs_e    = ifelse(x_right, xa - x, xb - x)

            # Chosen step and e update (NR: e ← step before clamping)
            @. d_chos  = ifelse(para_ok, d_para, $CGOLD * gs_e)

            # Edge guard: parabolic u too close to bracket endpoints → use ±tol1
            # Reuse para_ok as the edge-use mask
            @. para_ok = para_ok &
                         (((x + d_para) - xa < 2*tol1) | (xb - (x + d_para) < 2*tol1))
            @. d_chos  = ifelse(para_ok, ifelse(x < mid, tol1, -tol1), d_chos)

            # e ← chosen step before clamping (NR: e = d, pre-clamp)
            @. e       = d_chos

            # Enforce minimum step of tol1 away from x
            @. d       = ifelse(abs(d_chos) >= tol1, d_chos,
                                ifelse(d_chos >= zero(T), tol1, -tol1))

            # Evaluate g at trial point (mask converged lanes to avoid stale calls)
            @. u       = x + d
            @. u       = ifelse(converged, x, u)   # u_eval in-place
            g!(y, u, params...)
            @. y      = ifelse(converged, gx, y)  # mask converged result

            # ── Update bracket (gate on !converged) ──────────────────────────
            # NR rule (maximization, strict >):
            #   y>gx, u<x  → xb=x;  y>gx, u>=x → xa=x
            #   gu≤gx, u<x  → xa=u;  gu≤gx, u>=x → xb=u
            # XOR truth table verifies: update_a = (u<x) ⊻ (y>gx)
            # is_best reused as gu_best; x_right reused as u_lt_x then update_a
            @. is_best = y < gx
            @. d_para  = ifelse(is_best, x, u)      # bound (reuse d_para, done above)
            @. x_right = (u < x) ⊻ is_best          # update_a
            @. xa       = ifelse(converged, xa, ifelse(x_right, d_para, xa))
            @. xb       = ifelse(converged, xb, ifelse(x_right, xb, d_para))

            # ── Update best points x, w, v (gate on !converged) ──────────────
            @. v   = ifelse(converged, v,  ifelse(is_best, w,  v))
            @. gv  = ifelse(converged, gv, ifelse(is_best, gw, gv))
            @. w   = ifelse(converged, w,  ifelse(is_best, x,  w))
            @. gw  = ifelse(converged, gw, ifelse(is_best, gx, gw))
            @. x   = ifelse(converged, x,  ifelse(is_best, u,  x))
            @. gx  = ifelse(converged, gx, ifelse(is_best, y, gx))

            @. is_2nd = !is_best & ((y >= gw) | (w == x))
            @. v   = ifelse(converged, v,  ifelse(is_2nd, w,  v))
            @. gv  = ifelse(converged, gv, ifelse(is_2nd, gw, gv))
            @. w   = ifelse(converged, w,  ifelse(is_2nd, u,  w))
            @. gw  = ifelse(converged, gw, ifelse(is_2nd, y, gw))

            @. is_3rd = !is_best & !is_2nd & ((y >= gv) | (v == x) | (v == w))
            @. v   = ifelse(converged, v,  ifelse(is_3rd, u,  v))
            @. gv  = ifelse(converged, gv, ifelse(is_3rd, y, gv))

        end

        # Final convergence check
        @. converged = abs(x - (xa+xb)/2) <= 2*(tol*abs(x)+$eps(T)) - (xb-xa)/2

        return x, gx, converged
    end
end