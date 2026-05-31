package com.sistemasdistr.basico.repository;

import com.sistemasdistr.basico.model.SesionMuestreo;
import com.sistemasdistr.basico.model.User; 
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface SesionMuestreoRepository extends JpaRepository<SesionMuestreo, Integer> {
    
    List<SesionMuestreo> findByUsuarioOrderByFechaDesc(User usuario);
}