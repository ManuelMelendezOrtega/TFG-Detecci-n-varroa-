package com.tfg.basico.repository;

import com.tfg.basico.model.SesionMuestreo;
import com.tfg.basico.model.User; 
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;

@Repository
public interface SesionMuestreoRepository extends JpaRepository<SesionMuestreo, Integer> {
    
    List<SesionMuestreo> findByUsuarioOrderByFechaDesc(User usuario);

    List<SesionMuestreo> findByUsuarioAndFecha(User usuario, LocalDate fecha);
}