from clem_sim import cluster

if __name__ == "__main__":
    clstr = cluster(
        Nbh=21000,
        n_mus=1000,
        spin_dis="unif",
    )
    clstr.populate()
    clstr.build_mass_grid()
    clstr.emit_GW()
