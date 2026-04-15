#=

This entire file was written by iterating with Claude.

=#

function brent(g, x::AbstractArray, contexts...; xa, xb, tol=eltype(x)(1e-13), maxiter=500, check_every=1)
  # Brent doesn't use derivatives so parse either
  if contexts isa Tuple{Vararg{Context}}
    gc = DI.fix_tail(g, map(DI.unwrap, contexts)...)
    y = gc(x)
  else
    y = g(x, contexts...)
  end
  g!(_y, _x, _contexts...) = (_y .= g(_x, _contexts...); _y)
  return brent!(g!, y, copy(x), contexts...; xa, xb, tol, maxiter, check_every)
end

function brent!(
  g!::Function,
  y::AbstractArray,
  x::AbstractArray,
  contexts...;
  xa,
  xb,
  tol         = eltype(x)(1e-13),
  maxiter     = 500,
  check_every = 1,
)
  @assert length(xa) == length(xb)
  # Brent doesn't use derivatives so parse properly
  if contexts isa Tuple{Vararg{Context}}
    gc! = DI.fix_tail(g!, map(DI.unwrap, contexts)...)
  else
    gc! = (_y, _x)->g!(_y, _x, contexts...)
  end
  return _brent!(gc!, y, x; xa, xb, tol, maxiter, check_every)
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
@generated function _brent!(
  g!::Function,
  y,
  x::AbstractArray{T};
  xa,
  xb,
  tol         = T(1e-13),
  maxiter     = 500,
  check_every = 1,
  iters=similar(x, Int),
  retcode=similar(x, UInt8),
) where {T}
  # This weirdness (generated) is needed for compatibility with Metal
  # which literally cannot compile even if we immediately cast compile 
  # time constant to Float32.
  CGOLD = T(0.3819660112501051)
  return quote
    @assert length(xa) == length(xb) == length(x) == length(y)
    fill!(retcode, RETCODE_MAXITER)
    fill!(iters, -1)
    out = (; u=x, iters=iters, retcode=retcode)

    @. x  = xa + $CGOLD * (xb - xa)
    gx = similar(y)
    g!(y, x)
    w  = copy(x);  gw = copy(y)
    v  = copy(x);  gv = copy(y)

    # Algorithm state
    d         = similar(xa); d .= zero(T)   # last step taken
    e         = similar(xa); e .= zero(T)   # step before last (NR: start at 0)

    # ── Scratch arrays: 10 float + 5 bool ────────────────────────────────
    #
    # Float scratch — reuse strategy noted inline:
    #   p     : holds r=(x-w)*(y-gv) first, then p_final
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
    #   is_best : gx<y; reused as gu_best for bracket update
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
        out.iters .= ifelse.(
            (abs.(x .- mid) .<= 2 .* tol1 .- (xb .- xa) ./ 2) .&& out.iters .== -1, 
            iter-1, 
            out.iters
        )

        out.retcode .= ifelse.(out.iters .!= -1, RETCODE_SUCCESS, RETCODE_MAXITER)
        if all(out.retcode .== RETCODE_SUCCESS)
          return out
        end
      end
      # ── Parabolic interpolation ───────────────────────────────────────
      # Compute r into p, q_raw into q, then:
      #   q_final = 2*(q_raw - r)        [update q while p=r]
      #   p_final = (x-v)*q_final/2 + r*(w-v)  [algebraic rearrangement,
      #             avoids storing r separately; verified by substitution]
      @. p     = (x - w) * (y - gv)              # r → p
      @. q     = (x - v) * (y - gw)              # q_raw → q
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
      @. u       = ifelse(retcode .== RETCODE_SUCCESS, x, u)   # u_eval in-place
      g!(gx, u)
      @. gx      = ifelse(retcode .== RETCODE_SUCCESS, y, gx)  # mask retcode .== RETCODE_SUCCESS result

      # ── Update bracket (gate on !converged) ──────────────────────────
      # NR rule (maximization, strict >):
      #   gx>y, u<x  → xb=x;  gx>y, u>=x → xa=x
      #   gu≤gx, u<x  → xa=u;  gu≤gx, u>=x → xb=u
      # XOR truth table verifies: update_a = (u<x) ⊻ (gx>y)
      # is_best reused as gu_best; x_right reused as u_lt_x then update_a
      # Here we actually do minimization tho:
      @. is_best = gx < y
      @. d_para  = ifelse(is_best, x, u)      # bound (reuse d_para, done above)
      @. x_right = (u < x) ⊻ is_best          # update_a
      @. xa       = ifelse(retcode .== RETCODE_SUCCESS, xa, ifelse(x_right, d_para, xa))
      @. xb       = ifelse(retcode .== RETCODE_SUCCESS, xb, ifelse(x_right, xb, d_para))

      # ── Update best points x, w, v (gate on !converged) ──────────────
      @. v   = ifelse(retcode .== RETCODE_SUCCESS, v,  ifelse(is_best, w,  v))
      @. gv  = ifelse(retcode .== RETCODE_SUCCESS, gv, ifelse(is_best, gw, gv))
      @. w   = ifelse(retcode .== RETCODE_SUCCESS, w,  ifelse(is_best, x,  w))
      @. gw  = ifelse(retcode .== RETCODE_SUCCESS, gw, ifelse(is_best, y, gw))
      @. x   = ifelse(retcode .== RETCODE_SUCCESS, x,  ifelse(is_best, u,  x))
      @. y  = ifelse(retcode .== RETCODE_SUCCESS, y, ifelse(is_best, gx, y))

      @. is_2nd = @. is_2nd = !is_best & ((gx <= gw) | (w == x)) 
      @. v   = ifelse(retcode .== RETCODE_SUCCESS, v,  ifelse(is_2nd, w,  v))
      @. gv  = ifelse(retcode .== RETCODE_SUCCESS, gv, ifelse(is_2nd, gw, gv))
      @. w   = ifelse(retcode .== RETCODE_SUCCESS, w,  ifelse(is_2nd, u,  w))
      @. gw  = ifelse(retcode .== RETCODE_SUCCESS, gw, ifelse(is_2nd, gx, gw))

      @. is_3rd = @. is_3rd = !is_best & !is_2nd & ((gx <= gv) | (v == x) | (v == w)) 
      @. v   = ifelse(retcode .== RETCODE_SUCCESS, v,  ifelse(is_3rd, u,  v))
      @. gv  = ifelse(retcode .== RETCODE_SUCCESS, gv, ifelse(is_3rd, gx, gv))
    end

    # Final convergence check
    out.iters .= ifelse.(
        (abs.(x .- mid) .<= 2 .* tol1 .- (xb .- xa) ./ 2) .&& out.iters .== -1, 
        maxiter, 
        out.iters
    )

    out.retcode .= ifelse.(out.iters .!= -1, RETCODE_SUCCESS, RETCODE_MAXITER)
    return out
  end
end

#=




    # ====== #



    @assert length(xa) == length(xb)

    fill!(retcode, RETCODE_MAXITER)
    fill!(iters, -1)
    out = (; u=x, iters=iters, retcode=retcode)

    a  = copy(xa)
    b  = copy(xb)
#=
    x  = @. a + $CGOLD * (b - a)
    g!(y, x)
    w  = copy(x);  gw = copy(y)
    v  = copy(x);  gv = copy(y)
    =#
    # FIXED: evaluate at both golden-ratio points, keep the better one
x  = @. a + $CGOLD * (b - a)
g!(y, x)
x2 = @. b - $CGOLD * (b - a)   # = a + (1-CGOLD)*(b-a)
y2 = similar(y); g!(y2, x2)
# Keep whichever is lower
better = y2 .< y
@. x  = ifelse(better, x2, x)
@. y  = ifelse(better, y2, y)
w  = copy(x);  gw = copy(y)
v  = copy(x);  gv = copy(y)

    # Algorithm state
    d         = similar(a); d .= zero(T)   # last step taken
    e         = similar(a); e .= zero(T)   # step before last (NR: start at 0)

    # ── Scratch arrays: 10 float + 5 bool ────────────────────────────────
    #
    # Float scratch — reuse strategy noted inline:
    #   p     : holds r=(x-w)*(y-gv) first, then p_final
    #   q     : holds q_final = 2*(q_raw - r)
    #   d_para: parabolic step p/q; reused as bound in bracket update
    #   gs_e  : golden-section sub-interval (a-x or b-x)
    #   d_chos: chosen step pre-clamp; also written to e
    #   mid   : (a+b)/2; reused for tol2 expression inline
    #   tol1  : tol*|x|+eps(T); negated inline where -tol1 needed
    #   u     : trial point x+d; reused as u_eval (masked in-place)
    #   gu    : g(u_eval); masked in-place for converged lanes
    #
    # Bool scratch:
    #   para_ok : parabola accepted flag; reused for use_edge mask
    #   x_right : x>=mid; reused as u_lt_x then update_a
    #   is_best : gu>y; reused as gu_best for bracket update
    #   is_2nd, is_3rd: point bookkeeping
    mid     = similar(a)
    tol1    = similar(a)
    p       = similar(a)
    q       = similar(a)
    d_para  = similar(a)
    gs_e    = similar(a)
    d_chos  = similar(a)
    u       = similar(a)
    gu      = similar(a)
    para_ok = similar(a, Bool)
    x_right = similar(a, Bool)
    is_best = similar(a, Bool)
    is_2nd  = similar(a, Bool)
    is_3rd  = similar(a, Bool)

    for iter in 1:maxiter

      @. mid  = (a + b) / 2
      @. tol1 = tol * abs(x) + $eps(T)

      if check_every > 0 && mod(iter, check_every) == 0
        out.iters .= ifelse.(
            (abs.(x .- mid) .<= 2 .* tol1 .- (b .- a) ./ 2) .&& out.iters .== -1, 
            iter-1, 
            out.iters
        )

        out.retcode .= ifelse.(out.iters .!= -1, RETCODE_SUCCESS, RETCODE_MAXITER)
        if all(out.retcode .== RETCODE_SUCCESS)
          return out
        end
      end

      # ── Parabolic interpolation ───────────────────────────────────────
      # Compute r into p, q_raw into q, then:
      #   q_final = 2*(q_raw - r)        [update q while p=r]
      #   p_final = (x-v)*q_final/2 + r*(w-v)  [algebraic rearrangement,
      #             avoids storing r separately; verified by substitution]
      @. p     = (x - w) * (y - gv)              # r → p
      @. q     = (x - v) * (y - gw)              # q_raw → q
      @. q     = 2 * (q - p)                       # q_final (p=r still valid)
      @. p     = (x - v) * q / 2 + p * (w - v)   # p_final

      # Flip sign so q > 0 (NR convention), then abs(q)
      @. p     = ifelse(q > zero(T), -p, p)
      @. q     = abs(q)

      # e holds step from 2 iters ago (saved before overwrite)
      # para_ok temporarily holds the acceptance test result
      @. para_ok = (abs(e) > tol1) &
                    (abs(p) < abs(q) * abs(e / 2)) &
                    (p > q * (a - x)) &
                    (p < q * (b - x))

      # Parabolic step (guarded against q==0)
      @. d_para  = ifelse(q != zero(T), p / q, zero(T))

      # Golden-section step into the larger sub-interval
      @. x_right = x >= mid
      @. gs_e    = ifelse(x_right, a - x, b - x)

      # Chosen step and e update (NR: e ← step before clamping)
      @. d_chos  = ifelse(para_ok, d_para, $CGOLD * gs_e)

      # Edge guard: parabolic u too close to bracket endpoints → use ±tol1
      # Reuse para_ok as the edge-use mask
      @. para_ok = para_ok &
                    (((x + d_para) - a < 2*tol1) | (b - (x + d_para) < 2*tol1))
      @. d_chos  = ifelse(para_ok, ifelse(x < mid, tol1, -tol1), d_chos)

      # e ← chosen step before clamping (NR: e = d, pre-clamp)
      @. e       = d_chos

      # Enforce minimum step of tol1 away from x
      @. d       = ifelse(abs(d_chos) >= tol1, d_chos,
                          ifelse(d_chos >= zero(T), tol1, -tol1))

      # Evaluate g at trial point (mask converged lanes to avoid stale calls)
      @. u       = x + d
      @. u       = ifelse(out.retcode == RETCODE_SUCCESS, x, u)   # u_eval in-place
      g!(gu, u)
      @. gu      = ifelse(out.retcode == RETCODE_SUCCESS, y, gu)  # mask converged result

      # ── Update bracket (gate on !converged) ──────────────────────────
      # NR rule (maximization, strict >):
      #   gu>y, u<x  → b=x;  gu>y, u>=x → a=x
      #   gu≤gx, u<x  → a=u;  gu≤gx, u>=x → b=u
      # XOR truth table verifies: update_a = (u<x) ⊻ (gu>y)
      # is_best reused as gu_best; x_right reused as u_lt_x then update_a

      # NOTE: THIS IS ACTUALLY MINIMIZING! I CHANGED IT!
      @. is_best = gu < y
      @. d_para  = ifelse(is_best, x, u)      # bound (reuse d_para, done above)
      @. x_right = (u < x) ⊻ is_best          # update_a
      @. a       = ifelse(out.retcode == RETCODE_SUCCESS, a, ifelse(x_right, d_para, a))
      @. b       = ifelse(out.retcode == RETCODE_SUCCESS, b, ifelse(x_right, b, d_para))

      # ── Update best points x, w, v (gate on !converged) ──────────────
      @. v   = ifelse(out.retcode == RETCODE_SUCCESS, v,  ifelse(is_best, w,  v))
      @. gv  = ifelse(out.retcode == RETCODE_SUCCESS, gv, ifelse(is_best, gw, gv))
      @. w   = ifelse(out.retcode == RETCODE_SUCCESS, w,  ifelse(is_best, x,  w))
      @. gw  = ifelse(out.retcode == RETCODE_SUCCESS, gw, ifelse(is_best, y, gw))
      @. x   = ifelse(out.retcode == RETCODE_SUCCESS, x,  ifelse(is_best, u,  x))
      @. y  = ifelse(out.retcode == RETCODE_SUCCESS, y, ifelse(is_best, gu, y))

      @. is_2nd = !is_best & ((gu <= gw) | (w == x))
      @. v   = ifelse(out.retcode == RETCODE_SUCCESS, v,  ifelse(is_2nd, w,  v))
      @. gv  = ifelse(out.retcode == RETCODE_SUCCESS, gv, ifelse(is_2nd, gw, gv))
      @. w   = ifelse(out.retcode == RETCODE_SUCCESS, w,  ifelse(is_2nd, u,  w))
      @. gw  = ifelse(out.retcode == RETCODE_SUCCESS, gw, ifelse(is_2nd, gu, gw))

      @. is_3rd = !is_best & !is_2nd & ((gu <= gv) | (v == x) | (v == w))
      @. v   = ifelse(out.retcode == RETCODE_SUCCESS, v,  ifelse(is_3rd, u,  v))
      @. gv  = ifelse(out.retcode == RETCODE_SUCCESS, gv, ifelse(is_3rd, gu, gv))
    end

    # Final convergence check
    out.iters .= ifelse.(
        (abs.(x .- mid) .<= 2 .* tol1 .- (b .- a) ./ 2) .&& out.iters .== -1, 
        maxiter, 
        out.iters
    )

    out.retcode .= ifelse.(out.iters .!= -1, RETCODE_SUCCESS, RETCODE_MAXITER)
    return out
  end
end
=#