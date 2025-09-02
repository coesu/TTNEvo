using ITensors: SiteType, @SiteType_str, @OpName_str, @StateName_str

ITensors.space(::SiteType"a"; conserve_qns=false) = 1

ITensors.op(::OpName"Id", ::SiteType"a") = [1]
ITensors.op!(::OpName"Id", ::SiteType"a") = [1]
ITensors.state(::StateName"↑", ::SiteType"a") = [1]
ITensors.state(::StateName"↓", ::SiteType"a") = [1]
