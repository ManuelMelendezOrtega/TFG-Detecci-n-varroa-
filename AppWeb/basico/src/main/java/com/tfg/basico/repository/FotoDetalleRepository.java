package com.tfg.basico.repository;

import com.tfg.basico.model.FotoDetalle;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface FotoDetalleRepository extends JpaRepository<FotoDetalle, Integer> {
    

    boolean existsByRutaImagenMarcada(String rutaImagenMarcada);
    

    boolean existsByRutaImagenOriginal(String rutaImagenOriginal);
    
}