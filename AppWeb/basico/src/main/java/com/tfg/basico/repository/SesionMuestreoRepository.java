package com.tfg.basico.repository;

import com.tfg.basico.model.SesionMuestreo;
import com.tfg.basico.model.User; 
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.List;

/**
 * Interfaz para gestionar las consultas de la tabla "sesion_muestreo".
 */
@Repository
public interface SesionMuestreoRepository extends JpaRepository<SesionMuestreo, Integer> {
    
    // Obtiene todos los análisis de un apicultor concreto, ordenados desde el más nuevo al más antiguo
    List<SesionMuestreo> findByUsuarioOrderByFechaDesc(User usuario);

    // Busca si un apicultor específico hizo un análisis en una fecha exacta (clave para el control de los 21 días)
    List<SesionMuestreo> findByUsuarioAndFecha(User usuario, LocalDate fecha);
}