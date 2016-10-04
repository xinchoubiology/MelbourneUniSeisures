cufft1D <- function(x, inverse=FALSE)
{
  if(!is.loaded("cufft")) {
    dyn.load("cufft.so")
  }
  n <- length(x)
  rst <- .C("cufft",
            as.integer(n),
            as.integer(inverse),
            as.double(Re(x)),
            as.double(Im(x)),
            re=double(length=n),
            im=double(length=n))
  rst <- complex(real = rst[["re"]], imaginary = rst[["im"]])
  return(rst)
}